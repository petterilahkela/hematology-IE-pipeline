"""
merge_and_convert.py
====================
Merges the LoRA adapter with the Poro2 base model and converts to GGUF format for Ollama.

Run this AFTER finetune.py has completed successfully.

Usage:
    python3 merge_and_convert.py

Steps:
    1. Load base model + LoRA adapter
    2. Merge adapter into base model
    3. Save merged model (~16GB, temporary)
    4. Convert to GGUF Q4_K_M (~5GB, final)
    5. Print instructions for Ollama registration and cleanup
"""

import os
import subprocess
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS — change these paths to match your setup
# ──────────────────────────────────────────────────────────────────────────────

# Path to the original base model in HuggingFace format
BASE_MODEL_PATH = "/mnt/data/petteril/models/poro2-instruct-hf"

# Path to the LoRA adapter saved by finetune.py
ADAPTER_PATH = "/mnt/data/petteril/models/poro2-finetuned-v2"

# Where to save the merged model (temporary, ~16GB)
MERGED_PATH = "/mnt/data/petteril/models/poro2-finetuned-v2-merged"

# Where to save the final GGUF file (~5GB)
GGUF_PATH = "/mnt/data/petteril/models/poro2-finetuned-v2.gguf"

# Path to llama.cpp (clone it first if not present)
LLAMA_CPP_PATH = "/mnt/data/petteril/llama.cpp"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Check that adapter exists
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Merge and Convert — Poro2 Fine-tuned Model")
print("=" * 60)

if not os.path.exists(ADAPTER_PATH):
    print(f"\nERROR: Adapter not found at {ADAPTER_PATH}")
    print("Make sure finetune.py has completed successfully first.")
    sys.exit(1)

if not os.path.exists(BASE_MODEL_PATH):
    print(f"\nERROR: Base model not found at {BASE_MODEL_PATH}")
    print("Download it first with huggingface_hub.")
    sys.exit(1)

print(f"\nBase model: {BASE_MODEL_PATH}")
print(f"Adapter:    {ADAPTER_PATH}")
print(f"Merged:     {MERGED_PATH}")
print(f"GGUF:       {GGUF_PATH}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load base model and tokenizer
# ──────────────────────────────────────────────────────────────────────────────

print("\nLoading base model...")
print("(This may take a few minutes — loading ~16GB into memory)")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cpu",  # merge on CPU to avoid GPU memory issues
)

print("Base model loaded.")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load LoRA adapter and merge into base model
# ──────────────────────────────────────────────────────────────────────────────

print("\nLoading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging adapter into base model...")
print("(This may take a few minutes)")
model = model.merge_and_unload()

print("Merge complete.")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Save merged model
# ──────────────────────────────────────────────────────────────────────────────

print(f"\nSaving merged model to {MERGED_PATH} ...")
print("(~16GB — this will take a few minutes)")

os.makedirs(MERGED_PATH, exist_ok=True)
model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)

print("Merged model saved.")

# Free memory before conversion
del model
torch.cuda.empty_cache()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Convert to GGUF
# ──────────────────────────────────────────────────────────────────────────────

print(f"\nConverting to GGUF format...")
print(f"Output: {GGUF_PATH}")

convert_script = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")

if not os.path.exists(convert_script):
    print(f"\nERROR: llama.cpp not found at {LLAMA_CPP_PATH}")
    print("Clone it first:")
    print(f"  git clone https://github.com/ggerganov/llama.cpp {LLAMA_CPP_PATH}")
    print(f"  pip install -r {LLAMA_CPP_PATH}/requirements.txt")
    sys.exit(1)

# Step 1: Convert to f16
f16_path = GGUF_PATH + ".f16.gguf"
result = subprocess.run(
    [sys.executable, convert_script, MERGED_PATH, "--outtype", "f16", "--outfile", f16_path],
    check=False
)
if result.returncode != 0:
    print("\nERROR: F16 conversion failed.")
    sys.exit(1)
print("F16 conversion complete. Quantizing to Q4_K_M...")

# Step 2: Quantize to Q4_K_M
quantize_bin = os.path.join(LLAMA_CPP_PATH, "build", "bin", "llama-quantize")
result2 = subprocess.run(
    [quantize_bin, f16_path, GGUF_PATH, "Q4_K_M"],
    check=False
)
if result2.returncode != 0:
    print("\nERROR: Quantization failed.")
    print(f"F16 GGUF still available at: {f16_path}")
    sys.exit(1)

os.remove(f16_path)
print("GGUF Q4_K_M conversion complete.")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Done, print next steps
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
print()
print(f"✅ GGUF file created: {GGUF_PATH}")
print()
print("Next steps:")
print()
print("1. Register with Ollama:")
print(f"   echo 'FROM {GGUF_PATH}' > Modelfile")
print(f"   ollama create poro2-finetuned -f Modelfile")
print()
print("2. Test with pipeline:")
print("   python3 extract_hematology_prompting.py \\")
print("       --input /mnt/data/petteril/data/testi_50.xlsx \\")
print("       --output /mnt/data/petteril/data/results_poro2_ft.xlsx \\")
print("       --model poro2-finetuned \\")
print("       --batch-size 1 --concurrency 1")
print()
print("3. When you have confirmed everything works, delete the merged model:")
print(f"   rm -rf {MERGED_PATH}")
print(f"   (This will free ~16GB of disk space)")
print("=" * 60)
