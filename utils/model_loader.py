"""
utils/model_loader.py — Model Loading Utilities
Fake News Detection System | Ayush Deval

WHY THIS FILE EXISTS:
  ML models are large files (10 MB – 500 MB).  Loading them on every HTTP
  request would make each prediction take 5-30 seconds.  Instead, we load
  all models ONCE when Flask starts and keep them in a dictionary in RAM.
  Every subsequent request reads from that dictionary → millisecond latency.
"""

import os
import pickle
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _path(*parts: str) -> str:
    """Resolve a path relative to the models/ directory."""
    return os.path.join(MODELS_DIR, *parts)


# ──────────────────────────────────────────────
# INDIVIDUAL LOADERS
# ──────────────────────────────────────────────

def load_lr_model() -> dict:
    """
    Load Logistic Regression model + TF-IDF vectorizer from .pkl files.

    WHY .pkl:
        Pickle (pickle / joblib) serialises Python objects directly.
        Scikit-learn models are plain Python objects, so .pkl is the
        standard and fastest format.
    """
    logger.info("Loading Logistic Regression model …")
    with open(_path("lr_model.pkl"), "rb") as f:
        lr_model = pickle.load(f)
    with open(_path("tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    logger.info("LR model loaded ✓")
    return {"lr_model": lr_model, "tfidf": tfidf}


def load_lstm_model() -> dict:
    """
    Load LSTM Keras model (.h5) and its tokenizer (.pkl).

    WHY .h5 (HDF5):
        Keras uses HDF5 as its native save format.  It stores both
        the model architecture AND the trained weights in one file.
        load_model() reconstructs the exact network automatically.

    WHY separate tokenizer.pkl:
        The Keras tokenizer (word→integer mapping) is a Python object,
        not part of the network graph, so it must be saved/loaded
        separately with pickle.
    """
    logger.info("Loading LSTM model …")
    from keras.models import load_model  # lazy import — TF is heavy
    lstm_model = load_model(_path("lstm_model.h5"), compile=False)
    with open(_path("tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    logger.info("LSTM model loaded ✓")
    return {"lstm_model": lstm_model, "lstm_tokenizer": tokenizer}


def load_bert_model() -> dict:
    """
    Load DistilBERT model and tokenizer from the bert_model/ directory.

    WHY LAZY / OPTIONAL:
        The BERT model directory can be 250–500 MB.  On free deployment
        tiers (512 MB RAM), loading BERT alongside LR + LSTM may cause
        OOM errors.  We attempt to load it but fall back gracefully.

    WHY DistilBERT over full BERT:
        DistilBERT = 40 % smaller, 60 % faster, 97 % of BERT accuracy.
        Critical for free-tier deployment (Railway / Render / HF Spaces).
    """
    bert_dir = _path("bert_model")
    if not os.path.isdir(bert_dir):
        logger.warning("bert_model/ directory not found — BERT disabled.")
        return {}

    logger.info("Loading BERT (DistilBERT) model …")
    from transformers import pipeline  # lazy import
    bert_pipe = pipeline(
        "text-classification",
        model=bert_dir,
        device=-1,          # -1 = CPU; use 0 for GPU if available
        truncation=True,
        max_length=512,
    )
    logger.info("BERT model loaded ✓")
    return {"bert_pipe": bert_pipe}


# ──────────────────────────────────────────────
# MAIN LOADER — called once from app.py
# ──────────────────────────────────────────────

def load_all_models() -> dict:
    """
    Load every model and merge into a single cache dictionary.
    Keys available after loading:
        lr_model       — sklearn LogisticRegression
        tfidf          — sklearn TfidfVectorizer
        lstm_model     — keras Sequential
        lstm_tokenizer — keras Tokenizer
        bert_pipe      — transformers pipeline  (optional)
    """
    cache: dict = {}

    # Logistic Regression (fast, always load)
    try:
        cache.update(load_lr_model())
    except FileNotFoundError:
        logger.error("LR model files not found in models/. Run notebooks/02_train_lr.ipynb first.")

    # LSTM (medium, always load)
    try:
        cache.update(load_lstm_model())
    except FileNotFoundError:
        logger.error("LSTM model files not found. Run notebooks/03_train_lstm.ipynb first.")

    # BERT (large, optional — graceful fallback)
    try:
        cache.update(load_bert_model())
    except Exception as exc:
        logger.warning("BERT failed to load (%s). BERT endpoint will return 503.", exc)

    return cache