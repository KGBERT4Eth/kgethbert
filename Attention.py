import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
class MatrixVectorScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):

        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn

class MultiheadAttPoolLayer(nn.Module):
    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):

        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn

class DIVEAttentionLayer(nn.Module):
    def __init__(self, sent_dim, node_dim, num_heads=2):
        super(DIVEAttentionLayer, self).__init__()
        self.sent_dim = sent_dim
        self.node_dim = node_dim
        self.num_heads = num_heads


        self.node2sent_proj = nn.Linear(node_dim, sent_dim)
        self.sent2node_proj = nn.Linear(sent_dim, node_dim)

        self.pooler = MultiheadAttPoolLayer(num_heads, node_dim, sent_dim)
        self.co_attention = nn.MultiheadAttention(embed_dim=sent_dim, num_heads=num_heads)

        self.fc = nn.Sequential(
            nn.Linear(sent_dim + sent_dim, sent_dim),
            nn.ReLU(),
            nn.Linear(sent_dim, sent_dim)
        )

    def forward(self, hidden_states, X):
        bs, seq_len, _ = hidden_states.size()
        _, max_num_nodes, _ = X.size()

        pooled_seq, _ = self.pooler(X[:, 0, :], hidden_states)

        node_rep_proj = self.node2sent_proj(X)

        co_attention_seq, _ = self.co_attention(
            query=hidden_states.transpose(0, 1),
            key=node_rep_proj.transpose(0, 1),
            value=node_rep_proj.transpose(0, 1)
        )

        pooled_seq_expand = pooled_seq.unsqueeze(1).expand(-1, seq_len, -1)
        fused_seq = self.fc(torch.cat((pooled_seq_expand, co_attention_seq.transpose(0, 1)), dim=2))

        co_attention_node, _ = self.co_attention(
            query=node_rep_proj.transpose(0, 1),
            key = hidden_states.transpose(0, 1),
            value = hidden_states.transpose(0, 1)
        )

        fused_node = self.sent2node_proj(co_attention_node.transpose(0, 1))

        return fused_seq, fused_node