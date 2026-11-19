# Hematology Information Extraction Pipeline

A two-stage LLM-based information extraction pipeline for Finnish haematology bone marrow referrals and reports. The pipeline extracts six structured clinical variables using prompt engineering, fine-tuning, and retrieval-augmented generation (RAG).

## Extracted Variables

| Variable | Source | Description |
|----------|--------|-------------|
| `suspected_dg` | Referral | Suspected diagnosis |
| `prev_cur_dg` | Referral | Previous confirmed diagnosis |
| `prev_cur_dg_status` | Referral | Status of prior condition |
| `treatment` | Referral | Most recent relevant treatment |
| `cur_dg` | Report | Current confirmed diagnosis |
| `cur_dg_status` | Report | Status of current diagnosis |

## Models

All models run locally via [Ollama](https://ollama.com) on the shared GPU server (loska) used for training and inference (NVIDIA RTX A4000, 16GB VRAM). The fine-tuned LLMs are registered in Ollama's internal registry and can be viewed using the ollama list command. The /mnt/data/petteril/models/ directory contains the base GGUF files and the FinBERT checkpoint.

| Model | Model Identifier / Location |
|-------|-----------------------------|
| Llama 3.1 8B Instruct | `llama3.1:8b` (Ollama) |
| Poro2 8B Instruct | `Poro2-In:latest` (Ollama) |
| Poro2 + FT | `poro2-finetuned-v3:latest` (Ollama) |
| Poro2 + FT RAG | `poro2-finetuned-rag-v3:latest` (Ollama) |
| FinBERT | `TurkuNLP/bert-base-finnish-cased-v1` (fine-tuned, saved at `/mnt/data/petteril/models/bert-cur-dg-v1`) |

## Code Structure

```
Code/
├── Prompt/          # LLM extraction scripts
├── FT/              # Fine-tuning scripts (QLoRA)
├── BERT/            # FinBERT benchmark scripts
├── Eval/            # Evaluation and analysis scripts
Logs/                # Training logs
```

## Pipeline Overview
Some utility functions in the extraction scripts (e.g. benchmark mode) were developed during early iteration and are not used in the final pipeline.
### 1. LLM Extraction (Prompt/)
```bash
# Standard prompt
python extract_hematology_prompting_0308.py \
    --input data.xlsx --output results.xlsx \
    --model llama3.1:8b --batch-size 1 --concurrency 1

# RAG version
python extract_hematology_prompting_rag_v3.py \
    --input data.xlsx --output results.xlsx \
    --model poro2-finetuned-rag-v3:latest --batch-size 1 --concurrency 1
```

### 2. Fine-tuning (FT/)
```bash
python prepare_finetune_data_v3.py
python prepare_finetune_data_rag_v3.py
python finetune_v3.py
python finetune_rag_v3.py  # Fine-tuning for the RAG pipeline
python merge_and_convert.py  # convert to GGUF and quantise
```

### 3. FinBERT Benchmark (BERT/)
```bash
python prepare_bert_data.py --train train_v3.xlsx --val val_v3.xlsx --test test_v3.xlsx --output bert_data/
python finetune_bert.py
python predict_bert.py --model bert-cur-dg-v1 --test bert_test.csv --gold test_v3.xlsx --output bert_predictions.xlsx
python evaluate_bert.py --input bert_predictions.xlsx --output bert_analysis/
```

### 4. Evaluation (Eval/)
```bash
python prepare_for_evaluation_v3.py --test test_v3.xlsx --llama results_llama.xlsx ...
python analyse_cur_dg_v4.py --input eval.xlsx --mapping label_mapping_extended_eval.json --output cur_dg_analysis/
python analyse_other_fields_v4.py --input eval.xlsx --output other_fields_analysis/
```

## Data

Patient data is not included in this repository due to privacy constraints. The train/val/test split (2,377 / 316 / 357 samples) was obtained by matching record IDs with a collaborator’s parallel project rather than using an independent splitting script. split_data_v3_NotUsed.py contains an alternative splitting approach that does not reproduce the exact split used.

## Requirements

- Python 3.8+
- Ollama
- transformers, torch, datasets, scikit-learn, pandas, openpyxl, matplotlib
