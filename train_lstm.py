"""
train_lstm.py — Train Bidirectional LSTM on ISOT Dataset
Fake News Detection System | Ayush Deval

ISOT Dataset setup:
    Place True.csv and Fake.csv inside the data/ folder.

Run:  python train_lstm.py
"""

import os, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils.preprocess import clean_text

MAX_VOCAB=50_000; MAX_LEN=300; EMBED_DIM=128; LSTM_UNITS=128
BATCH_SIZE=64;    EPOCHS=15;   MODELS_DIR="models"
TRUE_PATH="data/True.csv";     FAKE_PATH="data/Fake.csv"

os.makedirs(MODELS_DIR, exist_ok=True)
tf.random.set_seed(42); np.random.seed(42)


def load_isot():
    print("Loading ISOT dataset …")
    true_df = pd.read_csv(TRUE_PATH); true_df["label"] = 0
    fake_df = pd.read_csv(FAKE_PATH); fake_df["label"] = 1
    df = pd.concat([true_df, fake_df], ignore_index=True).dropna(subset=["text"])
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Total: {len(df):,}  Real: {(df.label==0).sum():,}  Fake: {(df.label==1).sum():,}")
    print("Cleaning text …")
    df["clean"] = df["text"].apply(clean_text)
    return df


def main():
    df = load_isot()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean"].values, df["label"].values,
        test_size=0.2, random_state=42, stratify=df["label"].values)

    print("Fitting tokenizer …")
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_raw)
    vocab_size = min(len(tokenizer.word_index)+1, MAX_VOCAB)
    print(f"  Vocab size: {vocab_size:,}")

    def pad(texts):
        return pad_sequences(tokenizer.texts_to_sequences(texts),
                             maxlen=MAX_LEN, padding="post", truncating="post")

    X_train = pad(X_train_raw)
    X_test  = pad(X_test_raw)

    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2)),
        Dense(64, activation="relu"), Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    model.summary()

    model.fit(X_train, y_train, validation_split=0.1,
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(f"{MODELS_DIR}/lstm_best.h5", monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ], verbose=1)

    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*50}\n  Accuracy: {acc:.4f}  AUC: {auc:.4f}\n{'='*50}")
    y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=["REAL","FAKE"]))

    model.save(f"{MODELS_DIR}/lstm_model.h5")
    with open(f"{MODELS_DIR}/tokenizer.pkl","wb") as f: pickle.dump(tokenizer, f)
    print(f"\nSaved: {MODELS_DIR}/lstm_model.h5")
    print(f"Saved: {MODELS_DIR}/tokenizer.pkl")
    print("\nLSTM training complete!")

if __name__ == "__main__":
    main()