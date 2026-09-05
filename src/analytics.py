"""
Dataset analytics for the Dataset Analytics & Model Performance pages.
All numbers are computed live from the dataset / metrics file.
"""

import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "news_dataset.csv")
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset with friendly failure (returns empty DataFrame)."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def dataset_overview(df: pd.DataFrame) -> dict:
    """Counts and percentages of REAL vs FAKE articles."""
    if df.empty or "label" not in df.columns:
        return {"total": 0, "real": 0, "fake": 0,
                "real_pct": 0.0, "fake_pct": 0.0}
    labels = df["label"].astype(str).str.upper()
    total = int(len(labels))
    real = int((labels == "REAL").sum())
    fake = int((labels == "FAKE").sum())
    return {
        "total": total,
        "real": real,
        "fake": fake,
        "real_pct": round(real / total * 100, 1) if total else 0.0,
        "fake_pct": round(fake / total * 100, 1) if total else 0.0,
    }


def subject_distribution(df: pd.DataFrame) -> "pd.DataFrame":  # noqa: F821
    """Articles per subject/category, if the column exists."""
    if df.empty or "subject" not in df.columns or "label" not in df.columns:
        return pd.DataFrame()
    sub = (
        df.groupby([df["subject"].astype(str).str.title(), "label"])
        .size()
        .reset_index(name="count")
    )
    sub.columns = ["Subject", "Label", "Count"]
    return sub


def load_metrics() -> dict:
    """Load metrics.json produced by train_model.py ({} if missing)."""
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def confusion_values(cm) -> dict:
    """Turn a raw confusion matrix into named TP/TN/FP/FN counts.

    Class order from sklearn is alphabetical: ['FAKE', 'REAL'].
    Positive class for this project = FAKE.
    - True Positive: Actual FAKE, predicted FAKE (caught fake news)
    - False Negative: Actual FAKE, predicted REAL (missed fake news)
    - False Positive: Actual REAL, predicted FAKE (false alarm)
    - True Negative: Actual REAL, predicted REAL (correctly recognized real)
    """
    if not cm or len(cm) != 2:
        return {}
    tp = int(cm[0][0])  # actual FAKE, predicted FAKE
    fn = int(cm[0][1])  # actual FAKE, predicted REAL
    fp = int(cm[1][0])  # actual REAL, predicted FAKE
    tn = int(cm[1][1])  # actual REAL, predicted REAL
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "True Positive (actual FAKE, predicted FAKE)": tp,
        "False Negative (actual FAKE, predicted REAL)": fn,
        "False Positive (actual REAL, predicted FAKE)": fp,
        "True Negative (actual REAL, predicted REAL)": tn,
        # Keep backward-compatible keys if needed
        "matrix": [[tp, fn], [fp, tn]],
    }
