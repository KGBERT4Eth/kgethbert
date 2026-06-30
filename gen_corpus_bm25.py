import csv
import random
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from rank_bm25 import BM25Okapi,BM25L,BM25Plus
from transformers import BertTokenizer
from utils import load_pkl, save_pkl

# BM25
# datasetname="dev"
# account_texts = []
# account_indices = []
# with open(f'./Data/SPN/{datasetname}.tsv', 'r', encoding='utf-8') as file:
#     reader = csv.reader(file, delimiter='\t')
#     next(reader)
#     for row in reader:
#         account_indices.append(int(row[0]))
#         account_texts.append(row[2])
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# corpus = [tokenizer.tokenize(text) for text in account_texts]
#
# with open(f'Data/SPN/{datasetname}_corpus.pkl', 'wb') as pkl_file:
#     pickle.dump(corpus, pkl_file)
#
# bm25 = BM25Okapi(corpus)
#
# # mask
# # BM25
# doc_token_bm25_scores = {}  # indexBM25
#
# for doc_idx, text_tokens in tqdm(enumerate(corpus), total=len(corpus), desc="Processing BM25 scores"):
#     scores = []
#     # tokenBM25
#     for tokens in text_tokens:
#         score = bm25.get_batch_scores(tokens, [doc_idx])
#         scores.append(score)
#     # BM25
#     doc_token_bm25_scores[account_indices[doc_idx]] = scores
#
# with open(f'Data/SPN/{datasetname}_token_bm25_scores.pkl', 'wb') as pkl_file:
#     pickle.dump(doc_token_bm25_scores, pkl_file)

# BM25
datasetname = "dev"
account_texts = []
bm25_scores = []
account_triples_indices = []
account_indices = []

triples = load_pkl("triples.pkl")

with open(f'./Data/{datasetname}.tsv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='\t')
    next(reader)
    for row in reader:
        account_indices.append(int(row[0]))
        account_texts.append(row[2])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

corpus = [tokenizer.tokenize(text) for text in account_texts]

with open(f'Data/{datasetname}_corpus.pkl', 'wb') as pkl_file:
    pickle.dump(corpus, pkl_file)

# bm25 = BM25Okapi(corpus)
bm25=BM25L(corpus)
# bm25=BM25Plus(corpus)

# tokenBM25bm25_scores
for text_tokens in tqdm(corpus, total=len(corpus), desc="Processing BM25 scores"):
    scores = []
    for tokens in text_tokens:
        score = bm25.get_batch_scores(tokens, [corpus.index(text_tokens)])
        scores.append(score)
    bm25_scores.append(scores)

# 
for account_index in tqdm(account_indices, desc="Finding triples indices"):
    triples_indices = [i for i, triple in enumerate(triples) if account_index in triple]
    account_triples_indices.append(triples_indices)

with open(f'{datasetname}_list_token_bm25L_scores.pkl', 'wb') as pkl_file:
    pickle.dump(bm25_scores, pkl_file)
