"""
utils/predict.py — Model Inference Functions
Fake News Detection System | Ayush Deval

WHY THIS FILE EXISTS:
  Each of the three models has a different input format and output format.
  Centralising inference here keeps app.py clean and makes it easy to swap
  or add models later without touching the route logic.
"""

import numpy as np
import logging
from utils.preprocess import clean_text, clean_text_for_bert

logger = logging.getLogger(__name__)

MAX_LSTM_LEN = 300   # Must match the value used during LSTM training


# ──────────────────────────────────────────────
# LOGISTIC REGRESSION
# ──────────────────────────────────────────────

def predict_lr(text: str, cache: dict) -> tuple[int, float]:
    """
    Logistic Regression prediction.

    Pipeline:
        raw text → clean_text() → TF-IDF vectorize → LR.predict_proba()

    WHY predict_proba() instead of predict():
        predict() gives only 0/1.  predict_proba() gives [P(REAL), P(FAKE)],
        which lets us return a meaningful confidence score to the user.

    Returns:
        (prediction, confidence)
        prediction: 0 = Real, 1 = Fake
        confidence: probability of the predicted class (0.0 – 1.0)
    """
    lr_model = cache["lr_model"]
    tfidf    = cache["tfidf"]

    cleaned = clean_text(text)
    vec     = tfidf.transform([cleaned])          # shape: (1, vocab_size)

    pred    = int(lr_model.predict(vec)[0])       # 0 or 1
    proba   = lr_model.predict_proba(vec)[0]      # [P(0), P(1)]
    confidence = float(proba[pred])

    logger.info("LR prediction: %s  confidence: %.4f", "Fake" if pred else "Real", confidence)
    return pred, confidence


# ──────────────────────────────────────────────
# LSTM
# ──────────────────────────────────────────────

def predict_lstm(text: str, cache: dict) -> tuple[int, float]:
    """
    LSTM prediction.

    Pipeline:
        raw text → clean_text() → tokenize → pad_sequences → LSTM.predict()

    WHY pad_sequences():
        Neural networks process fixed-size tensors.  Sentences shorter than
        MAX_LSTM_LEN get zero-padded at the end ('post' padding).  Longer
        sentences are truncated.  MAX_LSTM_LEN must match training.

    WHY sigmoid output:
        The final Dense(1, activation='sigmoid') outputs a single float in
        [0, 1] representing P(Fake).  We threshold at 0.5 to get a class.
    """
    from keras.preprocessing.sequence import pad_sequences

    lstm_model = cache["lstm_model"]
    tokenizer  = cache["lstm_tokenizer"]

    cleaned    = clean_text(text)
    sequences  = tokenizer.texts_to_sequences([cleaned])          # list of int lists
    padded     = pad_sequences(sequences, maxlen=MAX_LSTM_LEN, padding="post")

    prob_fake  = float(lstm_model.predict(padded, verbose=0)[0][0])

    pred       = 1 if prob_fake >= 0.5 else 0
    confidence = prob_fake if pred == 1 else (1.0 - prob_fake)

    logger.info("LSTM prediction: %s  confidence: %.4f", "Fake" if pred else "Real", confidence)
    return pred, confidence


# ──────────────────────────────────────────────
# BERT
# ──────────────────────────────────────────────

def predict_bert(text: str, cache: dict) -> tuple[int, float]:
    """
    BERT (DistilBERT) prediction via HuggingFace pipeline.

    Pipeline:
        raw text → minimal clean → BertTokenizer (internal) → DistilBERT → softmax

    WHY raw text for BERT:
        BERT's tokenizer performs sub-word tokenisation (WordPiece).
        It handles capitalisation, punctuation, and stop words natively.
        Pre-removing them destroys the contextual signals BERT relies on.

    WHY truncation=True / max_length=512:
        BERT's positional embeddings are hard-capped at 512 tokens.
        Text beyond 512 tokens is silently truncated by the pipeline.

    Labels convention:
        Training used:  0 → REAL,  1 → FAKE  → label "LABEL_0" / "LABEL_1"
    """
    if "bert_pipe" not in cache:
        raise RuntimeError("BERT model is not loaded. Check models/bert_model/ directory.")

    bert_pipe = cache["bert_pipe"]
    cleaned   = clean_text_for_bert(text)

    result    = bert_pipe(cleaned)[0]             # {"label": "LABEL_1", "score": 0.97}
    pred      = 1 if result["label"] == "LABEL_1" else 0
    confidence = float(result["score"])

    logger.info("BERT prediction: %s  confidence: %.4f", "Fake" if pred else "Real", confidence)
    return pred, confidence


# ──────────────────────────────────────────────
# ROUTER — called by app.py
# ──────────────────────────────────────────────

def run_prediction(text: str, model_choice: str, cache: dict) -> tuple[int, float]:
    """
    Route the inference request to the correct model function.

    Args:
        text:         Raw user input string.
        model_choice: One of 'lr', 'lstm', 'bert'.
        cache:        Dictionary of loaded model objects.

    Returns:
        (prediction, confidence) — same contract as individual functions.
    """
    if model_choice == "lr":
        return predict_lr(text, cache)
    elif model_choice == "lstm":
        return predict_lstm(text, cache)
    elif model_choice == "bert":
        return predict_bert(text, cache)
    else:
        raise ValueError(f"Unknown model: {model_choice}")