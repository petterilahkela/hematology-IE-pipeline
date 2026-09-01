import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/mnt/data/petteril/models/poro2-instruct-hf')

for fname in [
    '/mnt/data/petteril/V3/Data/finetune_v3/train_v3.jsonl',
    '/mnt/data/petteril/V3/Data/finetune_v3/val_v3.jsonl',
    '/mnt/data/petteril/V3/Data/finetune_rag_v3/train_v3.jsonl',
    '/mnt/data/petteril/V3/Data/finetune_rag_v3/val_v3.jsonl',
]:
    lengths = []
    with open(fname) as f:
        for line in f:
            ex = json.loads(line)
            text = ' '.join([m['content'] for m in ex['messages']])
            lengths.append(len(tokenizer.encode(text)))
    print(f'{fname.split("/")[-2]}/{fname.split("/")[-1]}: max={max(lengths)}, over_3456={sum(1 for l in lengths if l > 3456)}')
