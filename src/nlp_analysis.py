"""
NLP text analysis: statistics, sentiment, keyword extraction and
sensational-language indicators. All values are computed from the text -
nothing is hard-coded.
"""

import re
from collections import Counter

import pandas as pd

from src.preprocessing import PUNCT_SET, RE_URL, sentences, tokenize_words, to_text

# ---------------------------------------------------------------------------
# Sentiment lexicon (small, self-contained, no external API)
# ---------------------------------------------------------------------------
POSITIVE_WORDS = {
    "good", "great", "excellent", "positive", "success", "successful",
    "improve", "improved", "improvement", "growth", "gain", "gains", "win",
    "won", "victory", "support", "supported", "help", "helped", "benefit",
    "benefits", "beneficial", "strong", "progress", "hope", "hopeful",
    "celebrate", "celebrated", "achievement", "approved", "praise", "praised",
    "best", "better", "happy", "welcome", "welcomed", "boost", "boosted",
    "confident", "confidence", "promising", "record", "agreement", "peace",
    "recovery", "safe", "safety", "trusted", "verified", "consistent",
}
NEGATIVE_WORDS = {
    "bad", "worse", "worst", "fail", "failed", "failure", "crisis", "risk",
    "danger", "dangerous", "threat", "threaten", "threatened", "attack",
    "attacked", "war", "conflict", "violence", "dead", "death", "deaths",
    "died", "injury", "injuries", "harm", "harmful", "damage", "damaged",
    "destroy", "destroyed", "loss", "lost", "decline", "declined", "drop",
    "dropped", "fall", "fell", "fear", "afraid", "angry", "anger", "hate",
    "scandal", "corrupt", "corruption", "fraud", "lie", "lied", "lies",
    "deceive", "deceived", "deception", "conspiracy", "secret",
    "hidden", "banned", "illegal", "warning", "warned", "shocking",
    "unbelievable", "furious", "exposed", "leaked",
}
NEUTRAL_SENTIMENT_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Sensational-language indicators
# ---------------------------------------------------------------------------
SENSATIONAL_WORDS = {
    "shocking", "bombshell", "breaking", "exposed", "unbelievable",
    "miracle", "secret", "banned", "conspiracy", "wake", "truth",
    "insane", "outrageous", "explosive", "horrifying", "jaw-dropping",
    "you-wont-believe", "beware", "warning", "censored", "leaked",
}

RE_MULTI_PUNCT = re.compile(r"[!?]{2,}")
RE_NUMBER = re.compile(r"\d")


# ---------------------------------------------------------------------------
# Basic text statistics
# ---------------------------------------------------------------------------
def text_statistics(title="", text="") -> dict:
    """Compute the KPI statistics for the NLP Analysis page."""
    body = to_text(text)
    full = f"{to_text(title)} {body}".strip()

    words = tokenize_words(full)
    sents = sentences(full)
    word_count = len(words)
    char_count = len(full)
    sent_count = len(sents)
    uppercase_words = [w for w in re.findall(r"[A-Za-z]+", full) if w.isupper() and len(w) > 1]

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sent_count,
        "avg_sentence_length": round(word_count / sent_count, 1) if sent_count else 0.0,
        "avg_word_length": round(sum(len(w) for w in words) / word_count, 2) if words else 0.0,
        "uppercase_words": len(uppercase_words),
        "exclamation_marks": full.count("!"),
        "question_marks": full.count("?"),
        "urls": len(RE_URL.findall(full)),
        "numbers": len(RE_NUMBER.findall(full)),
        "words": words,
        "sents": sents,
    }


# ---------------------------------------------------------------------------
# Sentiment analysis (lexicon-based)
# ---------------------------------------------------------------------------
def analyze_sentiment(text: str) -> dict:
    """Lexicon-based sentiment. Returns label, score (-1..1) and matched words."""
    words = tokenize_words(text)
    if not words:
        return {"label": "Neutral", "score": 0.0, "positive": [], "negative": []}
    pos = [w for w in words if w in POSITIVE_WORDS]
    neg = [w for w in words if w in NEGATIVE_WORDS]
    score = (len(pos) - len(neg)) / len(words)
    score = max(-1.0, min(1.0, score))
    if score > NEUTRAL_SENTIMENT_THRESHOLD:
        label = "Positive"
    elif score < -NEUTRAL_SENTIMENT_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"
    return {"label": label, "score": round(score, 3),
            "positive": sorted(set(pos)), "negative": sorted(set(neg))}


# ---------------------------------------------------------------------------
# Keyword extraction (frequency based, stopword-filtered)
# ---------------------------------------------------------------------------
def extract_keywords(text: str, top_n: int = 10):
    """Top keywords by frequency after stopword removal (via tokenize_words +
    the STOPWORDS filter used in clean_text)."""
    from src.preprocessing import STOPWORDS  # local import to get final list

    words = [w for w in tokenize_words(text)
             if w not in STOPWORDS and len(w) > 2]
    return Counter(words).most_common(top_n)


def word_frequencies(text: str, top_n: int = 20):
    return extract_keywords(text, top_n=top_n)


def sentence_lengths(text: str):
    """Word count per sentence - used for the distribution chart."""
    return [len(tokenize_words(s)) for s in sentences(text)]


# ---------------------------------------------------------------------------
# Sensational language indicators
# ---------------------------------------------------------------------------
def sensational_indicators(title="", text="") -> dict:
    """Linguistic flags often associated with sensational content.

    NOTE: these are NOT proof of misinformation - they are style signals only.
    """
    full = f"{to_text(title)} {to_text(text)}".strip()
    words = re.findall(r"[A-Za-z]+", full)
    lower_words = [w.lower() for w in words]

    exclam = full.count("!")
    questions = full.count("?")
    uppercase = [w for w in words if w.isupper() and len(w) > 2]
    multi_punct = RE_MULTI_PUNCT.findall(full)
    sens_words = [w for w in lower_words if w in SENSATIONAL_WORDS]

    flags = {
        "excessive_exclamations": exclam >= 2,
        "excessive_questions": questions >= 2,
        "heavy_capitals": len(uppercase) >= 2,
        "repeated_punctuation": len(multi_punct) > 0,
        "sensational_vocabulary": len(sens_words) >= 1,
    }
    triggered = sum(flags.values())

    # Simple 0-100 style intensity based on counts (transparent formula)
    intensity = min(
        100,
        exclam * 8 + questions * 4 + len(uppercase) * 6
        + len(multi_punct) * 10 + len(sens_words) * 7,
    )
    return {
        "flags": flags,
        "triggered": triggered,
        "exclamations": exclam,
        "questions": questions,
        "uppercase_words": uppercase[:15],
        "repeated_punctuation": multi_punct[:10],
        "sensational_words": sorted(set(sens_words)),
        "intensity": intensity,
    }


def kpi_dataframe(stats: dict) -> "pd.DataFrame":  # noqa: F821
    """Render the statistics dict as a small DataFrame for st.dataframe."""
    rows = [
        ("Word count", stats["word_count"]),
        ("Character count", stats["char_count"]),
        ("Sentence count", stats["sentence_count"]),
        ("Average sentence length (words)", stats["avg_sentence_length"]),
        ("Average word length (chars)", stats["avg_word_length"]),
        ("Uppercase words", stats["uppercase_words"]),
        ("Exclamation marks", stats["exclamation_marks"]),
        ("Question marks", stats["question_marks"]),
        ("URLs detected", stats["urls"]),
        ("Numbers detected", stats["numbers"]),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])
