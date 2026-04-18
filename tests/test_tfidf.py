from src.features.tfidf_features import build_tfidf_vectorizer


def test_tfidf_fit_transform():
    vec = build_tfidf_vectorizer({"ngram_range": [1, 2], "min_df": 1, "max_df": 1.0, "max_features": 100})
    X = vec.fit_transform(["alpha beta gamma", "alpha beta", "gamma delta"])
    assert X.shape[0] == 3
