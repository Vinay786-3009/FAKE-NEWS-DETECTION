"""
Text preprocessing utilities for the Fake News Detection & News
Credibility Analyzer.

Provides a robust `clean_text()` NLP pipeline:
lowercase -> remove HTML -> remove URLs -> remove punctuation ->
normalize spaces -> tokenize -> remove stopwords -> optional lemmatization.

The module is defensive: missing values, non-string input, empty strings and
very short text never raise. If NLTK corpora are unavailable, a built-in
stopword list and a simple regex tokenizer are used instead.
"""

import re
import string
import unicodedata

# ---------------------------------------------------------------------------
# Optional NLTK support (used when the corpora are available)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on environment
    import nltk
    for _pkg in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    _nltk_stopwords = set(stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()
    NLTK_AVAILABLE = True
except Exception:  # nltk not installed or corpora missing
    _nltk_stopwords = set()
    _lemmatizer = None
    NLTK_AVAILABLE = False

# Fallback stopword list (small, common English function words)
FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "because",
    "as", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "than", "too", "very", "can",
    "will", "just", "should", "now", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "this", "that",
    "these", "those", "it", "its", "he", "she", "they", "them", "their",
    "we", "us", "our", "you", "your", "i", "me", "my",
}

if _nltk_stopwords:
    STOPWORDS = _nltk_stopwords
else:
    STOPWORDS = FALLBACK_STOPWORDS

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"(https?://\S+|www\.\S+)")
RE_NON_WORD = re.compile(r"[^a-z\s]")
RE_EXTRA_SPACE = re.compile(r"\s+")
RE_TOKEN = re.compile(r"[a-z]+")

URL_PATTERN = RE_URL  # public alias used by the NLP analysis module


def to_text(value) -> str:
    """Coerce any input safely to a plain string ('' for missing/invalid)."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN check
        return ""
    return str(value)


def pd_is_nan(value) -> bool:
    """Minimal NaN test without importing pandas here."""
    return isinstance(value, float) and value != value


def remove_html(text: str) -> str:
    """Strip HTML tags and unescape common entities."""
    text = RE_HTML.sub(" ", text)
    text = (
        text.replace("&amp;", "&").replace("&lt;", "<")
        .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    )
    return text


def remove_urls(text: str) -> str:
    """Replace URLs with a space."""
    return RE_URL.sub(" ", text)


def clean_text(text, remove_stopwords: bool = True,
               lemmatize: bool = True) -> str:
    """Full cleaning pipeline. Always returns a (possibly empty) string.

    Steps
    -----
    1. Coerce to string / handle None & NaN
    2. Lowercase
    3. Remove HTML tags
    4. Remove URLs
    5. Normalize unicode characters
    6. Remove punctuation / keep letters only
    7. Collapse extra whitespace
    8. Tokenize
    9. Remove stopwords
    10. Optional lemmatization (NLTK) - falls back to no-op
    """
    if text is None:
        return ""
    text = str(text)
    if not text.strip():
        return ""

    text = text.lower()
    text = remove_html(text)
    text = remove_urls(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = RE_NON_WORD.sub(" ", text)
    text = RE_EXTRA_SPACE.sub(" ", text).strip()

    if not text:
        return ""

    # Tokenization (NLTK word_tokenize if available, else regex)
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = RE_TOKEN.findall(text)
    else:
        tokens = RE_TOKEN.findall(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    if lemmatize and _lemmatizer is not None:
        try:
            tokens = [_lemmatizer.lemmatize(t) for t in tokens]
        except Exception:
            pass

    return " ".join(tokens)


def tokenize_words(text: str):
    """Lightweight tokenizer for statistics (keeps punctuation out)."""
    return RE_TOKEN.findall(str(text).lower())


def combine_title_text(title, text) -> str:
    """Combine title + body for vectorization, handling missing values."""
    t1 = to_text(title).strip()
    t2 = to_text(text).strip()
    combined = f"{t1} {t2}".strip()
    return combined


def dataset_label_map(labels) -> "pd.Series":  # noqa: F821
    """Normalize assorted label spellings to REAL / FAKE."""
    import pandas as pd

    def norm(x):
        s = to_text(x).strip().lower()
        if s in {"1", "true", "real", "reliable", "legit", "genuine"}:
            return "REAL"
        if s in {"0", "fake", "false", "unreliable", "hoax", "bogus"}:
            return "FAKE"
        return ""

    return pd.Series(labels).map(norm)


def sentences(text: str):
    """Very light sentence splitter (no external dependency required)."""
    text = to_text(text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in (s.strip() for s in parts) if p]


PUNCT_SET = set(string.punctuation)
