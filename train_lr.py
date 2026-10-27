"""
train_lr.py — Train Logistic Regression on ISOT Dataset
Fake News Detection System | Ayush Deval

ISOT Dataset setup:
    Place True.csv and Fake.csv inside the data/ folder.
    Download from Kaggle: search "ISOT Fake News Dataset"

Run:  python train_lr.py
"""

import os, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.preprocess import clean_text

MODELS_DIR   = "models"
TRUE_PATH    = "data/True.csv"
FAKE_PATH    = "data/Fake.csv"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_isot():
    """
    Load ISOT dataset from two separate CSVs.
    True.csv  = real news from Reuters  → label 0
    Fake.csv  = fake news               → label 1

    WHY two files:
        ISOT ships real and fake news as separate files.
        We merge them and add labels before training.
    """
    print("Loading ISOT dataset …")
    true_df        = pd.read_csv(TRUE_PATH)
    fake_df        = pd.read_csv(FAKE_PATH)
    true_df["label"] = 0   # REAL
    fake_df["label"] = 1   # FAKE
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.dropna(subset=["text"])
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    print(f"  Total: {len(df):,}  |  Real: {(df.label==0).sum():,}  Fake: {(df.label==1).sum():,}")
    print("Cleaning text …")
    df["clean"] = df["text"].apply(clean_text)
    return df


def build_tfidf(X_train):
    print("Fitting TF-IDF vectorizer …")
    tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1,2),
                            min_df=2, max_df=0.95, sublinear_tf=True)
    tfidf.fit(X_train)
    return tfidf


def train_lr(X_vec, y):
    print("Training Logistic Regression …")
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                               class_weight="balanced", random_state=42)
    model.fit(X_vec, y)
    return model


def evaluate(model, tfidf, X_test, y_test):
    y_pred = model.predict(tfidf.transform(X_test))
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}\n  Accuracy: {acc:.4f} ({acc*100:.2f}%)\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["REAL","FAKE"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def save(model, tfidf):
    with open(f"{MODELS_DIR}/lr_model.pkl", "wb") as f: pickle.dump(model, f)
    with open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", "wb") as f: pickle.dump(tfidf, f)
    print(f"\nSaved: {MODELS_DIR}/lr_model.pkl")
    print(f"Saved: {MODELS_DIR}/tfidf_vectorizer.pkl")


def main():
    df = load_isot()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    tfidf = build_tfidf(X_train)
    model = train_lr(tfidf.transform(X_train), y_train)
    evaluate(model, tfidf, X_test, y_test)
    save(model, tfidf)
    print("\nLogistic Regression training complete!")

if __name__ == "__main__":
    main()