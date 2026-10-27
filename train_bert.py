"""
train_bert.py — Fine-tune DistilBERT on ISOT Dataset
Fake News Detection System | Ayush Deval

ISOT Dataset setup:
    Place True.csv and Fake.csv inside the data/ folder.

Run:  python train_bert.py
"""

import os, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset
from transformers import (DistilBertTokenizerFast,
    DistilBertForSequenceClassification, TrainingArguments,
    Trainer, EarlyStoppingCallback)
from utils.preprocess import clean_text_for_bert

MODEL_NAME="distilbert-base-uncased"; MAX_LEN=256; BATCH_SIZE=16
EPOCHS=3; MODELS_DIR="models/bert_model"
TRUE_PATH="data/True.csv"; FAKE_PATH="data/Fake.csv"

os.makedirs(MODELS_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.labels    = torch.tensor(labels, dtype=torch.long)
        self.encodings = tokenizer(list(texts), truncation=True,
                                   padding=True, max_length=MAX_LEN,
                                   return_tensors="pt")
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k,v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_isot():
    print("Loading ISOT dataset …")
    true_df = pd.read_csv(TRUE_PATH); true_df["label"] = 0
    fake_df = pd.read_csv(FAKE_PATH); fake_df["label"] = 1
    df = pd.concat([true_df, fake_df], ignore_index=True).dropna(subset=["text"])
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Total: {len(df):,}")
    df["clean"] = df["text"].apply(clean_text_for_bert)
    return df


def main():
    df = load_isot()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"].values, df["label"].values.astype(int),
        test_size=0.2, random_state=42, stratify=df["label"].values)

    tokenizer     = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    test_dataset  = NewsDataset(X_test,  y_test,  tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0:"REAL",1:"FAKE"}, label2id={"REAL":0,"FAKE":1})

    args = TrainingArguments(
        output_dir=MODELS_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE*2,
        evaluation_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        logging_steps=50, report_to="none",
        fp16=(DEVICE=="cuda"), seed=42)

    trainer = Trainer(model=model, args=args,
        train_dataset=train_dataset, eval_dataset=test_dataset,
        compute_metrics=lambda p: {"accuracy": accuracy_score(
            p.label_ids, np.argmax(p.predictions, axis=-1))},
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])

    print("\nFine-tuning DistilBERT …")
    trainer.train()

    y_pred = np.argmax(trainer.predict(test_dataset).predictions, axis=-1)
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["REAL","FAKE"]))

    trainer.save_model(MODELS_DIR)
    tokenizer.save_pretrained(MODELS_DIR)
    print(f"\nSaved to {MODELS_DIR}/")
    print("BERT training complete!")

if __name__ == "__main__":
    main()