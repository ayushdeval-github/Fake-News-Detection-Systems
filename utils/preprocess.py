"""
utils/preprocess.py — Text Cleaning & Preprocessing Pipeline
Fake News Detection System | Ayush Deval

WHY THIS FILE EXISTS:
  Raw news text contains HTML tags, URLs, punctuation, stop words, and
  capitalization noise that confuse ML models. This module produces
  a clean, normalised string that every model can work with.
"""

import re
import string
import nltk

# Download required NLTK data on first run
for pkg in ("stopwords", "punkt", "wordnet"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer, WordNetLemmatizer

STOP_WORDS  = set(stopwords.words("english"))
STEMMER     = PorterStemmer()
LEMMATIZER  = WordNetLemmatizer()


# ──────────────────────────────────────────────
# CORE CLEANING FUNCTION
# ──────────────────────────────────────────────

def clean_text(text: str, use_stemming: bool = False) -> str:
    """
    Full preprocessing pipeline used by Logistic Regression and LSTM.

    Steps:
        1. Lowercase          — 'Fake' == 'fake'
        2. Remove URLs        — links carry no semantic value
        3. Remove HTML tags   — strip web-scraping artefacts
        4. Remove punctuation — models don't need '!', '?', etc.
        5. Remove digits      — numbers rarely help fake-news detection
        6. Tokenise           — split into word list
        7. Remove stop words  — 'the', 'is', 'at' add noise
        8. Lemmatise / Stem   — 'running' → 'run'  (reduces vocab)

    Args:
        text:         Raw input string.
        use_stemming: If True use PorterStemmer; otherwise WordNetLemmatizer.

    Returns:
        Cleaned, space-joined token string.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 5. Remove digits
    text = re.sub(r"\d+", " ", text)

    # 6. Tokenise (simple split — fast, no punkt dependency for inference)
    tokens = text.split()

    # 7. Remove stop words and very short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # 8. Normalise
    if use_stemming:
        tokens = [STEMMER.stem(t) for t in tokens]
    else:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ──────────────────────────────────────────────
# BERT — raw text (minimal cleaning)
# ──────────────────────────────────────────────

def clean_text_for_bert(text: str, max_chars: int = 512) -> str:
    """
    BERT has its own internal tokeniser and handles stop words, casing, and
    sub-word splitting natively.  We only strip HTML and truncate.

    WHY DIFFERENT FROM clean_text():
        Removing stop words or stemming before feeding BERT destroys the
        contextual signals the model was trained to exploit.  BERT needs
        natural language, not a bag-of-tokens.
    """
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML only
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    return text[:max_chars]


# ──────────────────────────────────────────────
# HELPER UTILITIES
# ──────────────────────────────────────────────

def get_text_stats(text: str) -> dict:
    """Return basic statistics useful for logging and analysis."""
    words = text.split()
    return {
        "char_count":   len(text),
        "word_count":   len(words),
        "unique_words": len(set(w.lower() for w in words)),
        "avg_word_len": round(sum(len(w) for w in words) / max(len(words), 1), 2),
    }


def validate_input(text: str) -> tuple[bool, str]:
    """
    Returns (is_valid, error_message).
    Call this before preprocessing to catch bad input early.
    """
    if not text or not text.strip():
        return False, "Input text is empty."
    if len(text.strip()) < 10:
        return False, "Input text is too short (minimum 10 characters)."
    if len(text) > 10_000:
        return False, "Input text is too long (maximum 10,000 characters)."
    return True, ""