import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import torch

# ======================
# 2. 
# ======================
def euclidean_dist(a, b):
    """
     a  b 
    :
      a, b: (dim, )  numpy 
    :
      , 
    """
    return np.sqrt(np.sum(np.square(a - b)))


def cosine_dist(a, b):
    """
     a  b  = 1 - 
     1 - dot(a,b) / (norm(a)*norm(b)).
    :
      a, b: (dim, )  numpy 
    :
      ,  ()
    """
    return 1 - dot(a, b) / (norm(a) * norm(b))


def cosine_dist_multi(a, b):
    """
     a (1, dim)  b (N, dim) 
    -  -1 * cos_sim
    -  (1 - cos_sim)
    :
      a: (dim, ) shape
      b: (N, dim) shape
    :
      (N, ) , 
    """
    num = dot(a, b.T)  #  a  b 
    denom = norm(a) * norm(b, axis=1)
    res = num / denom  # 
    #  -1 * res ""
    #  (1 - res) 
    return -1 * res


def euclidean_dist_multi(a, b):
    """
     a (dim, )  b (N, dim) 
    :
      a: (dim, )
      b: (N, dim)
    :
      (N, ) , 
    """
    return np.sqrt(np.sum(np.square(b - a), axis=1))


# ======================
# 3. 
# ======================
def get_neighbors(X, idx, metric="cosine", include_idx_mask=[]):
    """
     X (), idx (),
     X , 
    -  include_idx_mask  mask 
    -  (idx) 
    :
      X: (N, dim) 
      idx: , 
      metric: str, "cosine"  "euclidean"
      include_idx_mask: list, 
    :
      indices, distances:
        ()
    """
    a = X[idx, :]  # 
    # 
    indices = list(range(X.shape[0]))

    #  metric 
    if metric == "cosine":
        dist = cosine_dist_multi(a, X)
    elif metric == "euclidean":
        dist = euclidean_dist_multi(a, X)
    else:
        raise ValueError("Distance Metric Error:  'cosine'  'euclidean'")

    # DataFrame, 
    df = pd.DataFrame(zip(indices, dist), columns=["idx", "dist"]).sort_values("dist")
    # 
    df = df[df["idx"] != idx]
    indices_sorted = list(df["idx"])
    distances_sorted = list(df["dist"])

    #  include_idx_mask
    if len(include_idx_mask) > 0:
        indices_tmp = []
        distances_tmp = []
        for i, candidate_idx in enumerate(indices_sorted):
            if candidate_idx in include_idx_mask:
                indices_tmp.append(candidate_idx)
                distances_tmp.append(distances_sorted[i])
        indices_sorted = indices_tmp
        distances_sorted = distances_tmp

    return indices_sorted, distances_sorted


def get_rank(X, query_idx, target_idx, metric, include_idx_mask=[]):
    """
     target_idx  query_idx 
    :
      X: (N, dim) 
      query_idx: 
      target_idx: 
      metric: 
      include_idx_mask: , mask
    :
      rank, dist, num_neighbors
        rank:  (1-based, 1, 2, ...)
        dist: target_idx  query_idx 
        num_neighbors: 
       target_idx ,  None, None, num_neighbors
    """
    indices, distances = get_neighbors(X, query_idx, metric, include_idx_mask)
    if len(indices) > 0 and target_idx in indices:
        #  target_idx  indices 
        trg_idx = indices.index(target_idx)
        return trg_idx + 1, distances[trg_idx], len(indices)
    else:
        return None, None, len(indices)


# ======================
# 4. ENS 
# ======================
def generate_pairs(ens_pairs, min_cnt=2, max_cnt=2, mirror=True):
    """
     ENS 
    -  ENS , , .
    - min_cnt, max_cnt:  [min_cnt, max_cnt]  ENS 
    - mirror=True:  (addr1, addr2) ,  (addr2, addr1)

    :
      ens_pairs: pd.DataFrame,  ["name", "address"] 
      min_cnt, max_cnt: int,  ENS 
      mirror: bool, 
    :
      address_pairs: list of [addr1, addr2] 
      all_ens_names:  ENS 
    """
    pairs = ens_pairs.copy()
    #  ENS 
    ens_counts = pairs["name"].value_counts()
    address_pairs = []
    all_ens_names = []
    ename2addresses = {}

    #  ENS 
    for idx, row in pairs.iterrows():
        ename = row["name"]
        addr = row["address"]
        if ename not in ename2addresses:
            ename2addresses[ename] = []
        ename2addresses[ename].append(addr)

    #  cnt  ENS  min_cnt <= cnt <= max_cnt 
    for cnt in range(min_cnt, max_cnt + 1):
        ens_names = list(ens_counts[ens_counts == cnt].index)
        all_ens_names += ens_names
        #  ENS 
        for ename in ens_names:
            addrs = ename2addresses[ename]
            # i,j
            for i in range(len(addrs)):
                for j in range(i + 1, len(addrs)):
                    addr1, addr2 = addrs[i], addrs[j]
                    address_pairs.append([addr1, addr2])
                    if mirror:
                        address_pairs.append([addr2, addr1])

    return address_pairs, all_ens_names


# ======================
# 5. 
# ======================
def load_embedding(path):

    if path is None:
        raise ValueError("init_checkpoint checkpoint")

    #  .npy 
    embeddings_path = f"{path}/embedding.npy"
    address_path = f"{path}/address.npy"

    embeddings = np.load(embeddings_path, allow_pickle=True)
    address_for_embedding = np.load(address_path, allow_pickle=True)

    #  address  embedding ()
    address_to_embedding = {}
    for i in range(len(address_for_embedding)):
        address = address_for_embedding[i]
        embedding = embeddings[i]
        if address not in address_to_embedding:
            address_to_embedding[address] = []
        address_to_embedding[address].append(embedding)

    address_list = []
    embedding_list = []
    for addr, embeds in address_to_embedding.items():
        address_list.append(addr)
        if len(embeds) > 1:
            embedding_list.append(np.mean(embeds, axis=0))
        else:
            embedding_list.append(embeds[0])

    X = np.array(np.squeeze(embedding_list))
    return X, address_list


# ======================
# 6. 
# ======================
def main():
    # 
    args = parse_args()

    #  ENS 
    ens_pairs = pd.read_csv(args.ens_dataset)
    # name max_ens_per_address 
    max_ens_per_address = 1
    num_ens_for_addr = ens_pairs.groupby("address")["name"].nunique().sort_values(ascending=False).reset_index()
    excluded = list(num_ens_for_addr[num_ens_for_addr["name"] > max_ens_per_address]["address"])
    ens_pairs = ens_pairs[~ens_pairs["address"].isin(excluded)]

    # 
    address_pairs, all_ens_names = generate_pairs(ens_pairs, max_cnt=args.max_cnt)

    # 
    X, address_list = load_embedding(args.ENS_embed_path)

    #  address 
    address_to_idx = {}
    idx_to_address = {}
    cnt = 0
    for address in address_list:
        address_to_idx[address] = cnt
        idx_to_address[cnt] = address
        cnt += 1

    #  address_pairs 
    idx_pairs = []
    failed_address = []
    for pair in address_pairs:
        addr1, addr2 = pair[0], pair[1]
        try:
            idx1 = address_to_idx[addr1]
            idx2 = address_to_idx[addr2]
            idx_pairs.append([idx1, idx2])
        except:
            #  address_to_idx 
            failed_address.append(addr1)
            failed_address.append(addr2)
            continue

    #  (src, dst)  ()
    ground_truth_euclidean_distance = []
    for pair in idx_pairs:
        src_id = pair[0]
        dst_id = pair[1]
        src_embedding = X[src_id]
        dst_embedding = X[dst_id]
        dist_val = euclidean_dist(src_embedding, dst_embedding)
        ground_truth_euclidean_distance.append(dist_val)

    print(">>>  ground_truth_euclidean_distance %d ." % len(ground_truth_euclidean_distance))
    print(">>> ...")

    # 
    records = []
    pbar = tqdm(total=len(idx_pairs), desc="Evaluating rank")
    for pair in idx_pairs:
        query_idx, target_idx = pair[1], pair[0]
        # :  pair[1]  query_idx, pair[0]  target_idx
        #  "target_idx  query_idx "
        rank, dist, num_set = get_rank(X, query_idx, target_idx, args.metric)
        # : query_idx, target_idx, rank, dist, num_set,  filter( none)
        records.append((query_idx, target_idx, rank, dist, num_set, "none"))
        pbar.update(1)
    pbar.close()

    #  DataFrame
    result = pd.DataFrame(records, columns=["query_idx", "target_idx", "rank", "dist", "set_size", "filter"])
    #  idx 
    result["query_addr"] = result["query_idx"].apply(lambda x: idx_to_address[x])
    result["target_addr"] = result["target_idx"].apply(lambda x: idx_to_address[x])
    #  result.drop(...)  result
    #  result : result = result.drop(["query_idx","target_idx"], axis=1)


    output_file = "Data/ENS/test_deanoy" + ".csv"
    #  CSV
    result.to_csv(output_file, index=False)
    print(f">>> : {output_file}")

def parse_args():
    """
     argparse 
     tensorflow flags
    """
    parser = argparse.ArgumentParser(description="PyTorch version of ENS de-anonymization evaluation.")
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--ens_dataset", type=str, default="")
    parser.add_argument("--ENS_embed_path", type=str, default="eth_emb")
    parser.add_argument("--max_cnt", type=int, default=2)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()