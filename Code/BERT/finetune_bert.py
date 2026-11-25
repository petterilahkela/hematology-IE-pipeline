"""
finetune_bert.py
================
Fine-tunes TurkuNLP/bert-base-finnish-cased-v1 for cur_dg sequence classification.

Prerequisites:
    pip install transformers datasets torch scikit-learn --break-system-packages

Usage:
    python finetune_bert.py

After training:
    - Best checkpoint selected by validation macro F1 (not loss), because
      cross-entropy loss is dominated by the majority class ("normaali")
    - Best model saved to OUTPUT_DIR
    - Use predict_bert.py to run inference on test set
"""

import json
import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME  = "TurkuNLP/bert-base-finnish-cased-v1"
DATA_DIR    = "/mnt/data/petteril/V3/Data/bert_data"
OUTPUT_DIR = "/mnt/data/petteril/models/bert-cur-dg-v2-eval-loss"

TRAIN_FILE  = os.path.join(DATA_DIR, "bert_train.csv")
VAL_FILE    = os.path.join(DATA_DIR, "bert_val.csv")
LABEL2ID    = os.path.join(DATA_DIR, "label2id.json")
ID2LABEL    = os.path.join(DATA_DIR, "id2label.json")

MAX_LENGTH  = 512   # BERT's maximum sequence length
BATCH_SIZE  = 16
EPOCHS      = 10
LEARNING_RATE = 2e-5
WARMUP_STEPS  = 75   # ~10% of total steps (2377/16 * 5 * 0.1)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Check GPU
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Checking GPU...")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load label mappings
# ──────────────────────────────────────────────────────────────────────────────

with open(LABEL2ID, encoding="utf-8") as f:
    label2id = json.load(f)
with open(ID2LABEL, encoding="utf-8") as f:
    id2label_raw = json.load(f)
    id2label = {int(k): v for k, v in id2label_raw.items()}

num_labels = len(label2id)
print(f"\nNumber of categories: {num_labels}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load tokenizer and model
# ──────────────────────────────────────────────────────────────────────────────

print(f"\nLoading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# BertForSequenceClassification adds a classification head automatically
# No QLoRA or LoRA needed — BERT is small enough to fine-tune fully
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
print("Model loaded.")
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Load and tokenise data
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8")
    df["label_id"] = df["label"].map(label2id)
    # Drop rows with labels not in training set
    missing = df["label_id"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} rows with unknown labels dropped from {path}")
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)
    return df

print("\nLoading data...")
train_df = load_csv(TRAIN_FILE)
val_df   = load_csv(VAL_FILE)
print(f"  Train: {len(train_df)} rows")
print(f"  Val:   {len(val_df)} rows")

def tokenise(df):
    dataset = Dataset.from_dict({
        "text":  df["text"].tolist(),
        "label": df["label_id"].tolist(),
    })
    def _tokenise(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
    return dataset.map(_tokenise, batched=True)

print("Tokenising...")
train_dataset = tokenise(train_df)
val_dataset   = tokenise(val_df)
print("  Done.")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    accuracy = (preds == labels).mean()
    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "accuracy": round(float(accuracy), 4),
    }

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Training arguments
# ──────────────────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=25,
    max_grad_norm=1.0,

    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,

    fp16=False,
    report_to="none",
)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — Train
# ──────────────────────────────────────────────────────────────────────────────

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60 + "\n")

trainer.train()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 — Save
# ──────────────────────────────────────────────────────────────────────────────

print(f"\nSaving best model to {OUTPUT_DIR} ...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save label mappings alongside model for easy inference
with open(os.path.join(OUTPUT_DIR, "label2id.json"), "w", encoding="utf-8") as f:
    json.dump(label2id, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUTPUT_DIR, "id2label.json"), "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in id2label.items()}, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("Training complete!")
print(f"Model saved to: {OUTPUT_DIR}")
print("Next step: run predict_bert.py to generate test set predictions")
print("=" * 60)