import torch
import torch.nn as nn

class CLSNodeAttentionLayer(nn.Module):

    def __init__(self, cls_dim, node_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.cls_dim = cls_dim
        self.node_dim = node_dim

        self.node_proj = nn.Linear(node_dim, cls_dim)
        self.cls_proj = nn.Linear(cls_dim, node_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=cls_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.fuse_cls = nn.Sequential(
            nn.Linear(cls_dim + cls_dim, cls_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fuse_node = nn.Sequential(
            nn.Linear(cls_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    @staticmethod
    def _build_cls_only_key_padding_mask(sentence_hidden, attention_mask=None):
        bs, seq_len, _ = sentence_hidden.size()
        mask = torch.ones(bs, seq_len, dtype=torch.bool, device=sentence_hidden.device)
        mask[:, 0] = False
        if attention_mask is not None:
            mask = mask | attention_mask.eq(0)
            mask[:, 0] = False
        return mask

    def forward(self, sentence_hidden, node_repr, attention_mask=None):

        cls_hidden = sentence_hidden[:, 0:1, :]
        cls_only_mask = self._build_cls_only_key_padding_mask(sentence_hidden, attention_mask)
        node_proj = self.node_proj(node_repr)

        fused_cls_attn, attn_weights = self.attn(
            query=cls_hidden,
            key=node_proj,
            value=node_proj
        )

        fused_cls = self.fuse_cls(torch.cat([cls_hidden, fused_cls_attn], dim=-1))

        node_attn, _ = self.attn(
            query=node_proj,
            key=sentence_hidden,
            value=sentence_hidden,
            key_padding_mask=cls_only_mask,
        )

        fused_nodes = self.fuse_node(node_proj+node_attn)
        fused_sentence = sentence_hidden.clone()
        fused_sentence[:, 0:1, :] = fused_cls

        return fused_sentence, fused_nodes
