"""
Prediction utilities: load the trained pipeline, classify new text, produce a
model-confidence-based credibility score and per-prediction explanations.
"""

import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_text, combine_title_text  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

MIN_TEXT_LENGTH = 40  # characters of cleaned text required for a prediction

_model = None
_vectorizer = None


class ModelNotTrainedError(RuntimeError):
    """Raised when the saved model/vectorizer cannot be loaded."""


def load_model():
    """Load (and cache) the trained model and vectorizer.
    
    Self-healing: if model files are missing or incompatible, automatically trains
    the pipeline on data/news_dataset.csv.
    """
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        need_train = not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH))
        if not need_train:
            try:
                _model = joblib.load(MODEL_PATH)
                _vectorizer = joblib.load(VECTORIZER_PATH)
            except Exception:
                need_train = True

        if need_train:
            try:
                from src.train_model import main as train_pipeline
                train_pipeline()
                _model = joblib.load(MODEL_PATH)
                _vectorizer = joblib.load(VECTORIZER_PATH)
            except Exception as e:
                raise ModelNotTrainedError(
                    f"Model could not be loaded or auto-trained: {e}"
                )
    return _model, _vectorizer


def _get_model_coefficients(model):
    """Safely extract feature weights from LogisticRegression or CalibratedClassifierCV."""
    if hasattr(model, "coef_"):
        return model.coef_[0]
    if hasattr(model, "calibrated_classifiers_"):
        coefs = []
        for est in model.calibrated_classifiers_:
            if hasattr(est, "coef_"):
                coefs.append(est.coef_[0])
            elif hasattr(est, "estimator") and hasattr(est.estimator, "coef_"):
                coefs.append(est.estimator.coef_[0])
            elif hasattr(est, "base_estimator") and hasattr(est.base_estimator, "coef_"):
                coefs.append(est.base_estimator.coef_[0])
        if coefs:
            return np.mean(coefs, axis=0)
    return None


def risk_level(score: float) -> str:
    """Map a credibility score (0-100) to a risk category."""
    if score <= 25:
        return "Very High Risk"
    if score <= 50:
        return "High Risk"
    if score <= 70:
        return "Moderate Risk"
    if score <= 85:
        return "Low Risk"
    return "Very Low Risk"


def credibility_score(confidence_fake: float) -> float:
    """Convert model confidence into a 0-100 credibility indicator.

    confidence_fake is P(FAKE) from the (calibrated) model. Credibility is
    simply (1 - P(FAKE)) * 100, i.e. the model's confidence that the content
    matches the REAL class. This is a model-confidence indicator, NOT an
    official credibility rating.
    """
    return round(float(1.0 - confidence_fake) * 100, 1)


def explain_prediction(clean_text_str: str, top_n: int = 10) -> dict:
    """Extract the most influential TF-IDF features for THIS prediction.

    In binary classification with classes ['FAKE', 'REAL']:
    - positive coef (classes_[1]) pushes towards REAL.
    - negative coef pushes towards FAKE (classes_[0]).
    """
    model, vectorizer = load_model()
    X = vectorizer.transform([clean_text_str])
    feature_names = np.array(vectorizer.get_feature_names_out())

    coef = _get_model_coefficients(model)
    if coef is None:
        return {
            "top_fake_words": [],
            "top_real_words": [],
            "local_fake": [],
            "local_real": [],
        }

    # In sklearn ['FAKE', 'REAL']: positive coef -> REAL, negative coef -> FAKE
    # Global top words:
    global_top_fake = feature_names[np.argsort(coef)[:top_n]].tolist()
    global_top_real = feature_names[np.argsort(coef)[-top_n:]][::-1].tolist()

    row = np.asarray(X.todense()).ravel()
    nz = np.flatnonzero(row)
    if nz.size == 0:
        return {
            "top_fake_words": global_top_fake,
            "top_real_words": global_top_real,
            "local_fake": [],
            "local_real": [],
        }

    # Local contributions:
    # contrib_real = coef * row (positive means pushes towards REAL)
    # contrib_fake = -coef * row (positive means pushes towards FAKE)
    contrib_real = coef * row
    contrib_fake = -contrib_real

    # Top words pushing towards FAKE
    fake_nz = [i for i in nz if contrib_fake[i] > 0]
    fake_nz_sorted = sorted(fake_nz, key=lambda i: contrib_fake[i], reverse=True)[:top_n]
    local_fake = [(feature_names[i], float(contrib_fake[i])) for i in fake_nz_sorted]

    # Top words pushing towards REAL
    real_nz = [i for i in nz if contrib_real[i] > 0]
    real_nz_sorted = sorted(real_nz, key=lambda i: contrib_real[i], reverse=True)[:top_n]
    local_real = [(feature_names[i], float(contrib_real[i])) for i in real_nz_sorted]

    return {
        "top_fake_words": global_top_fake,
        "top_real_words": global_top_real,
        "local_fake": local_fake,
        "local_real": local_real,
    }


def predict_news(title="", text="") -> dict:
    """Full analysis of one article. Never raises on bad input.

    Returns a dict with 'ok', 'error', 'prediction', 'confidence',
    'prediction_confidence', 'credibility', 'risk', 'clean', 'explanation'.
    """
    result = {
        "ok": False, "error": "", "prediction": "", "confidence": None,
        "prediction_confidence": None, "credibility": None, "risk": "",
        "clean": "", "explanation": None,
    }

    combined = combine_title_text(title, text)
    cleaned = clean_text(combined)
    if not cleaned:
        result["error"] = "Please enter some news text to analyze."
        return result
    if len(cleaned) < MIN_TEXT_LENGTH:
        result["error"] = (
            "The text is too short for a meaningful prediction. "
            "Please enter at least a full headline plus a few sentences."
        )
        return result

    try:
        model, vectorizer = load_model()
    except ModelNotTrainedError as exc:
        result["error"] = str(exc)
        return result

    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]

    confidence_fake = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        if "FAKE" in classes:
            confidence_fake = float(proba[classes.index("FAKE")])
        else:
            confidence_fake = float(proba.max())

    if pred == "FAKE":
        pred_label = "POTENTIALLY FAKE"
        pred_confidence = confidence_fake if confidence_fake is not None else 1.0
    else:
        pred_label = "REAL"
        pred_confidence = (1.0 - confidence_fake) if confidence_fake is not None else 1.0

    result.update({
        "ok": True,
        "prediction": pred_label,
        "confidence": confidence_fake,
        "prediction_confidence": pred_confidence,
        "credibility": credibility_score(confidence_fake) if confidence_fake is not None else None,
        "risk": risk_level(credibility_score(confidence_fake)) if confidence_fake is not None else "",
        "clean": cleaned,
        "explanation": explain_prediction(cleaned),
    })
    return result
    return result


def predict_batch(texts) -> "tuple":  # noqa: F821
    """Predict a list of cleaned strings. Returns (predictions, probabilities)."""
    model, vectorizer = load_model()
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        classes = list(model.classes_)
        p_fake = (probas[:, classes.index("FAKE")] if "FAKE" in classes
                  else probas.max(axis=1))
    else:
        p_fake = np.full(len(preds), np.nan)
    return preds, p_fake
