"""
TSV (, ) .npy 

:
1. TSV:
   - 1: "0x..."40+ ()
   - 2:  ()

2. :
   - address_{checkpoint_name}_{model_index}.npy
   - embedding_{checkpoint_name}_{model_index}.npy

:
   python generate_embeddings.py \
     --tsv_file path/to/your_data.tsv \
     --pretrained_model_path path/to/pretrained_bert \
     --checkpoint_name MyModel \
     --model_index 100000 \
     --output_dir ./inter_data
"""

import os
import argparse
import numpy as np
import torch
from transformers import BertTokenizer,BertModel
from eval_phish import BertClassifier
from tqdm import tqdm

torch.serialization.add_safe_globals([BertClassifier])
def main():
    parser = argparse.ArgumentParser(description="Generate address embeddings from a pretrained BERT model.")
    # 1) /
    parser.add_argument("--tsv_file", type=str)
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--model_path", type=str)

    parser.add_argument("--output_dir", type=str, default="testENS")

    # 2) /
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # ========== :  TSV  (, ) ========== #
    addresses = []
    texts = []
    with open(args.tsv_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            # 
            if line_idx == 0:
                continue
            cols = line.strip().split('\t')
            if len(cols) < 2:
                continue
            addr = cols[0]
            text = cols[1]
            addresses.append(addr)
            texts.append(text)
    print(f" {args.tsv_file}  {len(addresses)} ")

    # ========== :  BERT tokenizer   ========== #
    print(" BERT tokenizer  ...")
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    # model = BertModel.from_pretrained(args.pretrained_model_path)
    model = BertClassifier(
        pretrained_model_path=args.pretrained_model_path,
    )
    # model = torch.load(args.model_path)
    model.eval()
    model.to("cuda:3" if torch.cuda.is_available() else "cpu")

    # ========== :  ========== #
    # :  texts  batch tokenizer  model pooler_output 
    all_embeddings = []

    def get_batch(start, end):
        """ start, end batchtextstokenizer"""
        batch_texts = texts[start:end]
        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt'
        )
        return encoding

    device = next(model.parameters()).device

    num_samples = len(texts)
    batch_size = args.batch_size
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f" {num_batches} batchbatch = {batch_size}")

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc=""):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_encoding = get_batch(start_idx, end_idx)

            input_ids = batch_encoding["input_ids"].to(device)
            attention_mask = batch_encoding["attention_mask"].to(device)

            _,batch_embeddings = model(input_ids, attention_mask=attention_mask)
            # :
            #   outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
            #   outputs.pooler_output   : [batch_size, hidden_size] (CLS)
            # embedding
            # batch_embeddings = outputs.pooler_output  # [batch_size, hidden_size]

            # CPUnumpy
            batch_embeddings = batch_embeddings.cpu().numpy()

            all_embeddings.append(batch_embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(":", all_embeddings.shape)

    # ========== :  ========== #
    #   - address_{checkpoint_name}_{model_index}.npy
    #   - embedding_{checkpoint_name}_{model_index}.npy
    os.makedirs(args.output_dir, exist_ok=True)

    addr_npy_path = os.path.join(args.output_dir, f"address.npy")
    emb_npy_path = os.path.join(args.output_dir, f"embedding.npy")

    np.save(addr_npy_path, np.array(addresses, dtype=object))
    np.save(emb_npy_path, all_embeddings)

    print(f": {addr_npy_path}")
    print(f": {emb_npy_path}")


if __name__ == "__main__":
    main()
