"""
Train and compare fake-news classifiers, save the best pipeline.

Usage:  python src/train_model.py

- TF-IDF features (word 1-2 grams + char 3-5 grams)
- Logistic Regression and LinearSVC (with calibration so it can output a
  confidence score)
- Stratified, reproducible train/test split
- The model with the best F1 score is saved, along with the vectorizer and
  a metrics JSON file used by the Streamlit app.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_text, combine_title_text, dataset_label_map  # noqa: E402

RANDOM_STATE = 42
DATA_PATH = "data/news_dataset.csv"
MODEL_DIR = "models"

VECTORIZER_SETTINGS = dict(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
)


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run src/generate_dataset.py first."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")
    df["text"] = df["text"].fillna("")
    if "title" in df.columns:
        df["title"] = df["title"].fillna("")
    else:
        df["title"] = ""
    # Drop rows with no usable text and exact duplicates
    df = df[df["text"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=["title", "text"])
    df["label"] = dataset_label_map(df["label"])
    df = df[df["label"].isin(["REAL", "FAKE"])]
    return df.reset_index(drop=True)


def build_model_matrix(df: pd.DataFrame):
    """Clean text and build the combined title+text feature string."""
    df = df.copy()
    df["combined"] = [
        combine_title_text(t, b) for t, b in zip(df["title"], df["text"])
    ]
    df["clean"] = df["combined"].apply(clean_text)
    df = df[df["clean"].str.strip().str.len() > 0]
    return df


def evaluate(model, X_tr, y_tr, X_te, y_te, name: str) -> dict:
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    try:
        proba_ok = hasattr(model, "predict_proba")
    except Exception:
        proba_ok = False
    return {
        "name": name,
        "model": model,
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, pos_label="FAKE")),
        "recall": float(recall_score(y_te, pred, pos_label="FAKE")),
        "f1": float(f1_score(y_te, pred, pos_label="FAKE")),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "supports_proba": bool(proba_ok),
        "y_true": list(y_te),
        "y_pred": list(pred),
    }


def main() -> dict:
    df = load_dataset()
    df = build_model_matrix(df)
    print(f"Training samples after cleaning: {len(df)}")
    print(f"Class balance: {df['label'].value_counts().to_dict()}")

    vectorizer = TfidfVectorizer(**VECTORIZER_SETTINGS)
    X = vectorizer.fit_transform(df["clean"])
    y = df["label"].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_STATE
        ),
        "Linear SVM (LinearSVC)": CalibratedClassifierCV(
            LinearSVC(C=1.0, random_state=RANDOM_STATE), cv=5
        ),
    }

    results = [
        evaluate(m, X_tr, y_tr, X_te, y_te, name)
        for name, m in models.items()
    ]

    for r in results:
        print(f"{r['name']:<24} acc={r['accuracy']:.4f} "
              f"prec={r['precision']:.4f} rec={r['recall']:.4f} "
              f"f1={r['f1']:.4f}")

    best = max(results, key=lambda r: r["f1"])
    print(f"Best model by F1: {best['name']}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best["model"], os.path.join(MODEL_DIR, "fake_news_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

    comparison = [
        {k: r[k] for k in ("name", "accuracy", "precision", "recall", "f1")}
        for r in results
    ]
    metrics = {
        "best_model_name": best["name"],
        "best_model": {k: best[k] for k in
                       ("accuracy", "precision", "recall", "f1")},
        "confusion_matrix": best["confusion_matrix"],
        "supports_proba": best["supports_proba"],
        "comparison": comparison,
        "dataset": {
            "total_articles": int(len(load_dataset())),
            "train_samples": int(X_tr.shape[0]),
            "test_samples": int(X_te.shape[0]),
            "features": int(X.shape[1]),
            "class_balance": df["label"].value_counts().to_dict(),
        },
        "vectorizer_settings": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in VECTORIZER_SETTINGS.items()
        },
        "random_state": RANDOM_STATE,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved models/fake_news_model.pkl, tfidf_vectorizer.pkl, metrics.json")
    return metrics


if __name__ == "__main__":
    main()
