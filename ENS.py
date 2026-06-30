#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-  dean_all_ens_pairs.csv  (name, address)
-  train/dev/test TSV  (address, text)
- 
- () dean_all_ens_pairs.csv ()
:
  --task direct    : 
  --task finetune  : 
  --task linear    : 
"""

import os
import argparse
import datetime
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel


# ============== A. () ============== #
# def cosine_dist_multi(a, b):
#     """
#     a: (dim, )
#     b: (N, dim)
#      -1*cos_sim ()
#     """
#     from numpy import dot
#     from numpy.linalg import norm
#     num = dot(a, b.T)
#     denom = norm(a) * norm(b, axis=1)
#     res = num / denom
#     return -1 * res  # 

def cosine_dist_multi(a, b):
    """
    a: (dim, ) 
    b: (N, dim) 
    : cosine_similarity ()
    """
    #  Torch 
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.float32)

    # 
    cosine_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b, dim=1)  # (N, )

    #  ()
    return cosine_sim.numpy() if not isinstance(cosine_sim, torch.Tensor) else cosine_sim


def euclidean_dist_multi(a, b):
    return np.sqrt(np.sum((b - a) ** 2, axis=1))


def get_neighbors(X, idx, metric="cosine", include_idx_mask=[]):
    a = X[idx]
    if metric == "cosine":
        dist = cosine_dist_multi(a, X)
    elif metric == "euclidean":
        dist = euclidean_dist_multi(a, X)
    else:
        raise ValueError("Distance Metric Error")

    df = pd.DataFrame(zip(range(len(X)), dist), columns=["idx", "dist"]).sort_values("dist")
    df = df[df["idx"] != idx]  # 
    inds = list(df["idx"])
    dists = list(df["dist"])

    if len(include_idx_mask) > 0:
        tmp_i = []
        tmp_d = []
        for i, cidx in enumerate(inds):
            if cidx in include_idx_mask:
                tmp_i.append(cidx)
                tmp_d.append(dists[i])
        inds = tmp_i
        dists = tmp_d
    return inds, dists


def get_rank(X, query_idx, target_idx, metric="cosine", include_idx_mask=[]):
    inds, dists = get_neighbors(X, query_idx, metric, include_idx_mask)
    if target_idx in inds:
        pos = inds.index(target_idx)
        return pos + 1, dists[pos], len(inds)
    else:
        return None, None, len(inds)


def generate_pairs(ens_pairs, min_cnt=2, max_cnt=2, mirror=True):
    """
     [name,address] =>  [min_cnt, max_cnt] name 
    mirror=True  (addr1, addr2) & (addr2, addr1)
    """
    pairs = ens_pairs.copy()
    ens_counts = pairs["name"].value_counts()
    address_pairs = []
    ename2addresses = {}

    for i, row in pairs.iterrows():
        nm, ad = row["name"], row["address"]
        ename2addresses.setdefault(nm, []).append(ad)

    for cnt in range(min_cnt, max_cnt + 1):
        names_in_cnt = list(ens_counts[ens_counts == cnt].index)
        for nm in names_in_cnt:
            addrs = ename2addresses[nm]
            for i in range(len(addrs)):
                for j in range(i + 1, len(addrs)):
                    a1, a2 = addrs[i], addrs[j]
                    address_pairs.append([a1, a2])
                    if mirror:
                        address_pairs.append([a2, a1])
    return address_pairs


# ============== B.  "dean_all_ens_pairs.csv" + "tsv" ============== #
# 
#   1)  name ( dean_all_ens_pairs.csv )
#   2)  text( .tsv )
#  name (=)()

def load_ens_csv(ens_csv_path):
    """
     dean_all_ens_pairs.csv:  ["name","address"]
    : dict[address]=name
    """
    df = pd.read_csv(ens_csv_path)
    addr2name = {}
    for i, row in df.iterrows():
        nm, ad = row["name"], row["address"]
        addr2name[ad] = nm
    return addr2name


def load_tsv(tsv_path):
    """
     train/dev/test.tsv: [address, text]
    : dict[address]=text
    """
    ad2tx = {}
    with open(tsv_path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if idx == 0:  # 
                continue
            cols = line.strip().split('\t')
            if len(cols) < 2:
                continue
            ad, tx = cols[0], cols[1]
            ad2tx[ad] = tx
    return ad2tx


# ============== C.  (addr1, text1, addr2, text2, label) => PairsDataset ============= #
class PairItem:
    def __init__(self, ad1, tx1, ad2, tx2, label):
        self.addr1 = ad1
        self.text1 = tx1
        self.addr2 = ad2
        self.text2 = tx2
        self.label = label


class PairsDataset(Dataset):
    """
     "->(name,text)"  :
      - :  name
      - :  name
    """

    def __init__(self, address_list, addr2name, addr2text, tokenizer, max_len=128):
        """
        :param address_list: ( train)
        :param addr2name: dict[address]=name
        :param addr2text: dict[address]=text
        """
        self.address_list = address_list
        self.addr2name = addr2name
        self.addr2text = addr2text
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 1)  name->addresses
        self.name2addresses = {}
        for ad in address_list:
            nm = addr2name.get(ad, None)  #  dean_all_ens_pairs.csv, None
            if nm is not None:
                self.name2addresses.setdefault(nm, []).append(ad)
        # 2) self.addr_text
        self.addr_text = list(address_list)  # index
        # 3)  pairs
        self.pairs = self._make_pairs()

    def _make_pairs(self, neg_ratio=3):
        """
        ( name) ( name)
        :param neg_ratio:  ( 1 )
        """
        #  name, **2** name 
        valid_name_list = []
        for nm, adlist in self.name2addresses.items():
            if len(adlist) >= 2:
                valid_name_list.append(nm)

        pos_items = []
        neg_pairs = []
        all_addresses = self.address_list

        # ==========  ========== #
        #  valid name,  addresses 
        #  label=1
        for nm in valid_name_list:
            ads = self.name2addresses[nm]
            if len(ads) < 2:
                continue
            # 
            for i in range(len(ads)):
                for j in range(i + 1, len(ads)):
                    a1, a2 = ads[i], ads[j]
                    pos_items.append((a1, a2))  # 
        random.shuffle(pos_items)

        # ==========  ========== #
        # : , name 
        #  = neg_ratio * 
        neg_needed = len(pos_items) * neg_ratio
        neg_count = 0
        attempt = 0

        while neg_count < neg_needed and attempt < 10 * neg_needed:
            attempt += 1
            a1 = random.choice(all_addresses)
            a2 = random.choice(all_addresses)
            if a1 == a2:
                continue
            nm1 = self.addr2name.get(a1, None)
            nm2 = self.addr2name.get(a2, None)
            if nm1 is not None and nm2 is not None and nm1 != nm2:
                neg_pairs.append((a1, a2))  # 
                neg_count += 1

        # 
        pairs = []
        for (a1, a2) in pos_items:
            pairs.append((a1, a2, 1))  # 
        for (a1, a2) in neg_pairs:
            pairs.append((a1, a2, 0))  # 

        # 
        random.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (ad1, ad2, label) = self.pairs[idx]
        txt1 = self.addr2text.get(ad1, "")
        txt2 = self.addr2text.get(ad2, "")

        enc1 = self.tokenizer(
            txt1,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        enc2 = self.tokenizer(
            txt2,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            "input_ids_1": enc1["input_ids"].squeeze(0),
            "attn_mask_1": enc1["attention_mask"].squeeze(0),
            "input_ids_2": enc2["input_ids"].squeeze(0),
            "attn_mask_2": enc2["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }


# ============== D.  ============== #
class BertEmbeddingModel(nn.Module):
    def __init__(self, pretrained_model_path, freeze_bert=False, embed_dim=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        hidden_size = self.bert.config.hidden_size
        if embed_dim and embed_dim < hidden_size:
            self.reduce = nn.Linear(hidden_size, embed_dim)
            self.outdim = embed_dim
        else:
            self.reduce = None
            self.outdim = hidden_size

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        if self.reduce:
            pooled = self.reduce(pooled)
        return pooled


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        dist = torch.norm(emb1 - emb2, dim=1)
        loss_pos = label * (dist ** 2)
        loss_neg = (1 - label) * (torch.relu(self.margin - dist) ** 2)
        return torch.mean(loss_pos + loss_neg)

# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=0.5):
#         super().__init__()
#         self.margin = margin
#
#     def forward(self, emb1, emb2, label):
#         # 
#         cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
#         loss_pos = label * (1 - cosine_sim)  #  1
#         loss_neg = (1 - label) * torch.relu(cosine_sim - self.margin)  #  margin
#         return torch.mean(loss_pos + loss_neg)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        i1 = batch["input_ids_1"].to(device)
        m1 = batch["attn_mask_1"].to(device)
        i2 = batch["input_ids_2"].to(device)
        m2 = batch["attn_mask_2"].to(device)
        lb = batch["label"].to(device)

        optimizer.zero_grad()
        emb1 = model(i1, m1)
        emb2 = model(i2, m2)
        loss = criterion(emb1, emb2, lb)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for batch in tqdm(loader, desc="eval", leave=False):
        i1 = batch["input_ids_1"].to(device)
        m1 = batch["attn_mask_1"].to(device)
        i2 = batch["input_ids_2"].to(device)
        m2 = batch["attn_mask_2"].to(device)
        lb = batch["label"].to(device)
        emb1 = model(i1, m1)
        emb2 = model(i2, m2)
        loss = criterion(emb1, emb2, lb)
        total += loss.item()
    return total / len(loader)


# ============== E. :  dean_all_ens_pairs.csv  ============== #
@torch.no_grad()
def ens_eval_dean(model, tokenizer, ens_csv_path, device, metric="euclidean", max_cnt=2, out_csv="ens_rank.csv"):
    """
    1) dean_all_ens_pairs.csv => [name, address]
    2) address->text => ??? 
       CSV 'text'  merge =>  'text'
    3) generate_pairs => address
    4) modelembedding => get_rank => 
    """
    df = pd.read_csv(ens_csv_path)
    if "text" not in df.columns:
        print("[Error] dean_all_ens_pairs.csv  text  (address->text) ")
        return {}
    # 
    max_ens_per_address = 1
    num_ens_for_addr = df.groupby("address")["name"].nunique().sort_values(ascending=False).reset_index()
    exclude = list(num_ens_for_addr[num_ens_for_addr["name"] > max_ens_per_address]["address"])
    df = df[~df["address"].isin(exclude)]
    df = df.dropna(subset=["text"])

    # 
    addr_pairs = generate_pairs(df[["name", "address"]], min_cnt=2, max_cnt=max_cnt, mirror=True)
    #  address->text
    addr2txt = dict(zip(df["address"], df["text"]))
    # address_list
    address_list = list(df["address"].unique())
    # build addr->idx
    addr2idx = {}
    idx2addr = {}
    for i, ad in enumerate(address_list):
        addr2idx[ad] = i
        idx2addr[i] = ad
    # batch encode => embedding
    texts = []
    for ad in address_list:
        texts.append(addr2txt.get(ad, ""))
    model.eval()

    batch_size = 8
    n = len(address_list)
    n_batch = (n + batch_size - 1) // batch_size
    emb_list = []
    for i in range(n_batch):
        s = i * batch_size
        e = min(s + batch_size, n)
        sub_texts = texts[s:e]
        encoding = tokenizer(
            sub_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids = encoding["input_ids"].to(device)
        attn_mask = encoding["attention_mask"].to(device)
        emb = model(input_ids, attn_mask)
        emb_np = emb.cpu().numpy()
        emb_list.append(emb_np)
    X = np.concatenate(emb_list, axis=0)  # (N, dim)

    # pairs => idx
    idx_pairs = []
    for p in addr_pairs:
        a1, a2 = p
        if (a1 in addr2idx) and (a2 in addr2idx):
            idx_pairs.append([addr2idx[a1], addr2idx[a2]])

    records = []
    pbar = tqdm(total=len(idx_pairs), desc="ENS Ranking")
    for pair in idx_pairs:
        q_idx, t_idx = pair[1], pair[0]
        rank, dist, num_set = get_rank(X, q_idx, t_idx, metric=metric)
        records.append((q_idx, t_idx, rank, dist, num_set))
        pbar.update(1)
    pbar.close()

    df_res = pd.DataFrame(records, columns=["query_idx", "target_idx", "rank", "dist", "set_size"])
    df_res["query_addr"] = df_res["query_idx"].apply(lambda x: idx2addr[x])
    df_res["target_addr"] = df_res["target_idx"].apply(lambda x: idx2addr[x])

    df_res.to_csv(out_csv, index=False)
    print(f"[ens_eval_dean]  {out_csv}")

    valid = df_res.dropna(subset=["rank"])
    if len(valid) > 0:
        avg_rank = valid["rank"].mean()
        hits1 = sum(valid["rank"] == 1) / len(valid)
        hits3 = sum(valid["rank"] <= 3) / len(valid)
        res = {"avg_rank": avg_rank, "hits@1": hits1, "hits@3": hits3}
        print("[ens_eval_dean] :", res)
        return res
    else:
        print("[ens_eval_dean] No valid pairs in ranking!")
        return {}


# ============== F.  ============== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="linear", choices=["direct", "finetune", "linear"])
    parser.add_argument("--ens_csv", type=str, default="")
    parser.add_argument("--train_tsv", type=str, default="gen_dean/ENS_sentence")
    parser.add_argument("--dev_tsv", type=str, default="gen_dean/ENS_sentence")
    parser.add_argument("--test_tsv", type=str, default="gen_dean/ENS_sentence")
    parser.add_argument("--max_cnt", type=int, default=2)
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"])

    parser.add_argument("--pretrained_model_path", type=str, default="Train_output/01_09_17_47")
    # parser.add_argument("--model_path", type=str, default="Eval_ENS/finetune/01_13_18_03/best_model.pth")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)#256
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dim", type=int, default=512)
    args = parser.parse_args()

    # 
    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    run_dir = f"Eval_ENS/{args.task}/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # 
    cfg_path = os.path.join(run_dir, "config.txt")
    with open(cfg_path, "w") as f:
        f.write(str(args) + "\n")

    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    # with open(cfg_path, "a") as f:
    #     f.write(f"Device: {device}\n")

    #  ens_csv => addr->name
    addr2name = load_ens_csv(args.ens_csv)

    #  direct dean_all_ens_pairs.csv 
    # =>  "text" ? :  CSV  text => ?
    # :
    #   1)  CSV,  text
    #   2)  test.tsv ranking,  "" 
    # : "direct"  test_tsv  "->text"  ens_csv
    #  ens_csv  "text"
    # :  test_tsv
    #   ->  train+dev+test 
    # ()

    def merge_text_into_ensdf(ensdf, ad2txt):
        # ensdf text
        #  address  ad2txt 
        # 
        txt_list = []
        for i, row in ensdf.iterrows():
            ad = row["address"]
            t = ad2txt.get(ad, "")
            txt_list.append(t)
        ensdf["text"] = txt_list
        return ensdf

    # tokenizer & model
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    freeze_bert = (args.task == "linear")
    model = BertEmbeddingModel(args.pretrained_model_path, freeze_bert=freeze_bert, embed_dim=args.output_dim)

    device_ids = [1, 2, 3]
    torch.cuda.set_device(device_ids[0])  # 
    if len(device_ids) > 1:
        print(f"Using GPUs: {device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device_ids[0])

    criterion = ContrastiveLoss(margin=1.0)

    if args.task == "direct":
        if args.model_path:
            model.load_state_dict(torch.load(args.model_path))
        #  => 
        # 1)  test.tsv => address->text
        # 2)  ens_csv =>  "text"
        ad2tx_test = load_tsv(args.test_tsv)
        ensdf = pd.read_csv(args.ens_csv)  # [name, address]
        ensdf = merge_text_into_ensdf(ensdf, ad2tx_test)
        # 3)  ens_eval_dean
        # out_csv= os.path.join(run_dir,"ens_ranking.csv")
        # res= ens_eval_dean(model, tokenizer, out_csv= out_csv, ens_csv_path= None,
        #                    device=device, metric=args.metric, max_cnt=args.max_cnt)
        # :  None =>  df. :
        # hack:   ens_eval_dean  df ?
        # : ensdf csv => ens_eval_dean
        temp_csv = os.path.join(run_dir, "merged_ens_text.csv")
        ensdf.to_csv(temp_csv, index=False)
        res = ens_eval_dean(model, tokenizer, ens_csv_path=temp_csv, device=device_ids[0],
                            metric=args.metric, max_cnt=args.max_cnt,
                            out_csv=os.path.join(run_dir, "ens_ranking.csv"))
        # 
        with open(os.path.join(run_dir, "result.txt"), "w") as f:
            f.write("[DIRECT] Evaluate on dean_all_ens_pairs:\n")
            f.write(str(res) + "\n")


    else:
        # finetune/linear => 
        #  train.tsv, dev.tsv =>  PairDataset
        # 1)  train/dev tsv => address->text
        ad2tx_train = load_tsv(args.train_tsv)
        ad2tx_dev = load_tsv(args.dev_tsv)

        #  train_dataset  dev_dataset

        train_addrs = list(ad2tx_train.keys())
        dev_addrs = list(ad2tx_dev.keys())
        train_dataset = PairsDataset(train_addrs, addr2name, ad2tx_train, tokenizer, max_len=args.max_length)
        dev_dataset = PairsDataset(dev_addrs, addr2name, ad2tx_dev, tokenizer, max_len=args.max_length)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

        #  BERT 

        if args.task == "linear":
            #  BERT 
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 10)
        else:
            # finetune 
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        best_val_loss = float('inf')
        best_train_loss = float('inf')
        best_model_path = os.path.join(run_dir, "best_val_model.pth")
        best_train_model_path = os.path.join(run_dir, "best_train_model.pth")
        train_losses = []
        val_losses = []

        for epoch in tqdm(range(1, args.epochs + 1), desc="Training Progress", unit="epoch"):

            tr_loss = train_epoch(model, train_loader, optimizer, criterion, device_ids[0])
            vl_loss = eval_epoch(model, dev_loader, criterion, device_ids[0])
            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            print(f"[Epoch {epoch}] train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}")

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved, val_loss={best_val_loss:.4f}")

            if tr_loss < best_train_loss:
                best_train_loss = tr_loss
                torch.save(model.state_dict(), best_train_model_path)
                print(f"Best train model saved, train_loss={best_train_loss:.4f}")

        #  => 
        model.load_state_dict(torch.load(best_model_path))
        #  dean_all_ens_pairs.csv + test.tsv  =>  => ranking
        ad2tx_test = load_tsv(args.test_tsv)
        ensdf = pd.read_csv(args.ens_csv)
        # 

        ensdf = merge_text_into_ensdf(ensdf, ad2tx_test)
        #  CSV
        temp_csv = os.path.join(run_dir, "merged_ens_text.csv")
        ensdf.to_csv(temp_csv, index=False)
        # 

        out_csv = os.path.join(run_dir, "ens_ranking.csv")
        res = ens_eval_dean(model, tokenizer, ens_csv_path=temp_csv, device=device_ids[0],
                            metric=args.metric, max_cnt=args.max_cnt, out_csv=out_csv)
        # 

        result_txt = os.path.join(run_dir, "result.txt")
        with open(result_txt, "w") as f:
            f.write(f"[{args.task}] Final result after training\n")
            f.write(f"Best val_loss: {best_val_loss:.4f}\n")
            f.write("Ranking Evaluate:\n")
            f.write(str(res) + "\n")

        # 
        model.load_state_dict(torch.load(best_train_model_path))
        res_train = ens_eval_dean(model, tokenizer, ens_csv_path=temp_csv, device=device_ids[0],
                                  metric=args.metric, max_cnt=args.max_cnt, out_csv=out_csv)
        # 
        with open(result_txt, "a") as f:
            f.write(f"Best train_loss: {best_train_loss:.4f}\n")
            f.write("Ranking Evaluate on training set:\n")
            f.write(str(res_train) + "\n")


if __name__ == "__main__":
    main()
