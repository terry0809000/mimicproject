import pandas as pd

from src.data.schemas import DatasetSchema, dataset_contract_check


def test_dataset_contract_check_passes():
    notes = pd.DataFrame({"row_id": [1], "subject_id": [1], "text": ["abc"]})
    ann = pd.DataFrame({"row_id": [1], "behavior_tobacco": [0]})
    schema = DatasetSchema(note_id_col="row_id", patient_id_col="subject_id", note_text_col="text", annotation_task_columns=["behavior_tobacco"])
    summary = dataset_contract_check(notes, ann, schema)
    assert summary["n_notes"] == 1
