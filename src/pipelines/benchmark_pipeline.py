from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV

from src.data.loaders import label_prevalence_table, link_notes_annotations, load_annotations, load_notes
from src.data.splitting import patient_level_split, save_splits, split_diagnostics
from src.evaluation.confusion import save_confusion_matrix
from src.evaluation.metrics import bootstrap_metric, classification_report_dict, per_class_table
from src.features.tfidf_features import build_tfidf_vectorizer, save_vectorizer
from src.models.rule_based import NegationAwareRuleModel, RuleBasedConfig
from src.training.trainers import cost_record, persist_model, train_sklearn_with_val
from src.preprocessing.section_extraction import apply_section_extraction
from src.preprocessing.text_cleaning import normalize_text
from src.reporting.export_tables import export_table
from src.utils.logging_utils import save_json


def prepare_dataset(cfg: dict) -> pd.DataFrame:
    notes = load_notes(
        cfg["paths"]["notes_path"],
        category_col=cfg["data"].get("category_col"),
        allowed_categories=cfg["data"].get("allowed_note_categories"),
    )
    annotations = load_annotations(cfg["paths"]["annotations_path"])
    merged, summary = link_notes_annotations(notes, annotations, cfg)
    text_col = cfg["data"]["note_text_col"]
    merged[text_col] = merged[text_col].astype(str).map(normalize_text)
    merged, qa = apply_section_extraction(merged, text_col, cfg["section_extraction"])

    out_dir = Path(cfg["paths"]["output_dir"])
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    merged.to_csv(cfg["paths"]["processed_path"], index=False)
    qa.to_csv(out_dir / "tables" / "section_extraction_qa.csv", index=False)
    save_json(summary, out_dir / "tables" / "linkage_summary.json")

    prev = label_prevalence_table(merged, cfg["data"]["annotation_task_columns"])
    export_table(prev, out_dir / "tables" / "label_prevalence")
    return merged


def make_splits(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    split_df = patient_level_split(df, cfg["data"]["patient_id_col"], seed=cfg["project"]["seed"])
    save_splits(split_df, cfg["paths"]["split_dir"], cfg["data"]["note_id_col"])
    diag = split_diagnostics(split_df, cfg["data"]["patient_id_col"])
    save_json(diag, Path(cfg["paths"]["output_dir"]) / "tables" / "split_diagnostics.json")
    return split_df


def train_rule_based_task(cfg: dict, df: pd.DataFrame, task: str) -> dict:
    lex = yaml.safe_load(Path(cfg["models"]["rule_based"]["lexicon_path"]).read_text())
    rb_cfg = RuleBasedConfig(lexicons=lex, negation_cues=lex.get("negation_cues", []), window_tokens=cfg["models"]["rule_based"]["window_tokens"])
    model = NegationAwareRuleModel(rb_cfg, task=task)
    test = df[df["split"] == "test"]
    y_true = test[task].to_numpy()
    y_pred = np.array(model.predict(test["analysis_text"].tolist()))
    metrics = classification_report_dict(y_true, y_pred)
    metrics.update(bootstrap_metric(y_true, y_pred, n_iter=cfg["evaluation"]["bootstrap_iterations"], seed=cfg["project"]["seed"]))
    return metrics


def train_tfidf_model_task(cfg: dict, df: pd.DataFrame, task: str, model_name: str) -> tuple[dict, pd.DataFrame]:
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]

    vec = build_tfidf_vectorizer(cfg["models"]["tfidf"])
    X_train = vec.fit_transform(tr["analysis_text"])
    X_val = vec.transform(va["analysis_text"])
    X_test = vec.transform(te["analysis_text"])

    y_train = tr[task].to_numpy()
    y_val = va[task].to_numpy()
    y_test = te[task].to_numpy()

    model, best_params, cost = train_sklearn_with_val(model_name, X_train, y_train, X_val, y_val, cfg["models"][model_name], cfg["project"]["seed"])
    if model_name == "linear_svm":
        model = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = classification_report_dict(y_test, y_pred, y_prob)
    metrics.update({f"best_{k}": v for k, v in best_params.items()})
    per_class = per_class_table(y_test, y_pred)

    root = Path(cfg["paths"]["output_dir"])
    save_vectorizer(vec, root / "models" / f"tfidf_{model_name}_{task}_vectorizer.joblib")
    persist_model(model, root / "models" / f"tfidf_{model_name}_{task}.joblib")
    save_confusion_matrix(y_test, y_pred, root / "figures" / f"cm_{model_name}_{task}.png", title=f"{model_name} {task}")

    cost_df = cost_record(model_name, task, "test", cost)
    return metrics, per_class.join(cost_df, how="cross")
