"""
finetune_v3.py
==============
QLoRA fine-tuning script for Poro 2 Instruct on the Finnish hematology extraction task.

Prerequisites:
    pip install transformers peft trl bitsandbytes accelerate --break-system-packages

Usage:
    python3 finetune_v3.py

    # Or run in background overnight:
    nohup python3 finetune_v3.py > Logs/finetune_log_v3.txt 2>&1 &

After training:
    - Best model is saved automatically to OUTPUT_DIR
    - Convert to GGUF with merge_and_convert.py for use with Ollama
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS — change these paths to match your setup
# ──────────────────────────────────────────────────────────────────────────────

# Path to the base model in HuggingFace format (get this from J)
MODEL_PATH = "/mnt/data/petteril/models/poro2-instruct-hf"

OUTPUT_DIR = "/mnt/data/petteril/models/poro2-finetuned-v3"
TRAIN_FILE = "/mnt/data/petteril/V3/Data/finetune_v3/train_v3.jsonl"
VAL_FILE   = "/mnt/data/petteril/V3/Data/finetune_v3/val_v3.jsonl"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Check GPU availability
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Checking GPU...")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print("=" * 60)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load tokenizer and model
# ──────────────────────────────────────────────────────────────────────────────

print(f"\nLoading model from {MODEL_PATH} ...")

# Load the tokenizer (converts text to numbers the model understands)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Add padding token if missing (needed for batch processing)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# QLoRA: load model in 4-bit to reduce GPU memory during training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},  
)


print("Model loaded successfully.")


# Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Configure LoRA
# LoRA trains only a small fraction of the model's parameters
# instead of the full model, making fine-tuning feasible on limited GPU
# ──────────────────────────────────────────────────────────────────────────────

lora_config = LoraConfig(
    r=16,                      # Rank: how many parameters to train (higher = more capacity)
    lora_alpha=32,             # Scaling factor (usually 2x rank)
    target_modules=[           # Which layers to fine-tune (attention layers)
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ],
    lora_dropout=0.05,         # Dropout to prevent overfitting
    bias="none",
    task_type="CAUSAL_LM",     # Causal language modeling (text generation)
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print how many parameters are being trained
model.print_trainable_parameters()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Load training data
# ──────────────────────────────────────────────────────────────────────────────

print(f"\nLoading training data...")
train_dataset = load_dataset("json", data_files=TRAIN_FILE)["train"]
val_dataset   = load_dataset("json", data_files=VAL_FILE)["train"]

print(f"  Train: {len(train_dataset)} examples")
print(f"  Val:   {len(val_dataset)} examples")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Configure training
# ──────────────────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Number of passes through the entire training data
    # Early stopping will stop earlier if needed
    num_train_epochs=5,
    # How many examples to process at once on each GPU
    # Keep low (2) to fit in 16GB VRAM
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    # Accumulate gradients over 4 steps before updating
    # Effectively gives batch size of 2 * 4 = 8 without extra memory
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    # Learning rate — how fast the model adjusts its weights
    learning_rate=2e-4,

    # Use float16 to save GPU memory
    fp16=True,

    # Evaluate validation loss every 200 steps
    eval_steps=200,
    evaluation_strategy="steps",

    # Save model checkpoint every 200 steps
    save_steps=200,
    save_strategy="steps",

    # Log training progress every 25 steps
    logging_steps=25,

    # Automatically load the best checkpoint at the end
    # (the one with lowest validation loss)
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Warm up learning rate gradually for first 100 steps
    warmup_steps=100,

    # Don't send data to external services (Weights & Biases etc.)
    report_to="none",
)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Create trainer with early stopping
# Early stopping automatically stops training when validation loss
# stops improving, preventing overfitting
# ──────────────────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,

    # Maximum sequence length — referrals + reports can be long
    max_seq_length=3456,

    # Stop training if validation loss doesn't improve for 2 evaluations
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — Train the model
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Starting training...")
print("Monitor training_log.txt for progress if running in background")
print("=" * 60 + "\n")

trainer.train()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 — Save the final model
# ──────────────────────────────────────────────────────────────────────────────

print(f"\nSaving best model to {OUTPUT_DIR} ...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 60)
print("Training complete!")
print(f"Model saved to: {OUTPUT_DIR}")
print()
print("Next steps:")
print("  1. Convert to GGUF:")
print("     python3 llama.cpp/convert_hf_to_gguf.py " + OUTPUT_DIR + " --outtype q4_k_m")
print("  2. Create Ollama model:")
print("     echo 'FROM " + OUTPUT_DIR + ".gguf' > Modelfile")
print("     ollama create poro2-finetuned -f Modelfile")
print("  3. Test with pipeline:")
print("     python3 extract_hematology_prompting.py --model poro2-finetuned ...")
print("=" * 60)
