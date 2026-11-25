"""
sort_by_length.py
=================
Utility script to sort JSONL fine-tuning files by example length (longest first).

Reads each file, computes the token length of each example, sorts the examples
in descending order by length, and rewrites the file in-place. Used during
data preparation to inspect the longest examples and verify they stay within
the maximum sequence length.

Usage:
    python3 sort_by_length.py
"""

import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/mnt/data/petteril/models/poro2-instruct-hf')

for fname in [
    '/mnt/data/petteril/V3/Data/finetune_v3/train_v3.jsonl',
    '/mnt/data/petteril/V3/Data/finetune_rag_v3/train_v3.jsonl',
]:
    examples = []
    with open(fname) as f:
        for line in f:
            ex = json.loads(line)
            text = ' '.join([m['content'] for m in ex['messages']])
            length = len(tokenizer.encode(text))
            examples.append((length, line))
    examples.sort(key=lambda x: x[0], reverse=True)
    print(f'Longest: {examples[0][0]} tokens — {fname.split("/")[-2]}')
    with open(fname, 'w') as f:
        for length, line in examples:
            f.write(line)
    print(f'  Sorted: {fname}')
