"""
paper_tower/model.py
--------------------
PaperTower: SeHGNN-derived encoder that maps pre-propagated metapath features
into a fixed-dim L2-normalised embedding suitable for contrastive retrieval.

Inputs (at forward time):
    feat_dict : dict[str -> Tensor[B, nfeat]]
        Keys must match feat_keys supplied at construction time (e.g. "P", "PP", "PCP").
        Values are already row-normalised, propagated features — NO adjacency matrices
        needed here; propagation was done offline in step4_assemble_and_propagate.py.

Output:
    Tensor[B, embed_dim], L2-normalised.

Architecture (kept from SeHGNN hgb/model.py):
    LinearPerMetapath  — per-metapath MLP projection  [B, M, nfeat] → [B, M, hidden]
    Transformer        — cross-metapath semantic fusion
    fc_after_concat    — flatten + project             [B, M*hidden] → [B, hidden]

Added for LCR:
    proj_head          — Linear(hidden, embed_dim)
    L2 normalisation   — F.normalize(..., dim=-1)

Removed from SeHGNN:
    task_mlp           — classification head (wrong task)
    labels_embeding    — label propagation (no node labels in LCR)
    embeding           — node-type embedding lookup (we receive dense tensors directly)
    dataset / nclass / tgt_type arguments
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xavier_uniform_(tensor, gain=1.0):
    """Xavier uniform init that works for 3-D weight tensors (LinearPerMetapath)."""
    fan_in, fan_out = tensor.size()[-2:]
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return torch.nn.init._no_grad_uniform_(tensor, -a, a)


def _unfold_nested_list(x):
    return sum(x, [])


# ---------------------------------------------------------------------------
# Sub-modules (unchanged from SeHGNN)
# ---------------------------------------------------------------------------

class LinearPerMetapath(nn.Module):
    """Independent linear projection for each metapath channel."""

    def __init__(self, cin: int, cout: int, num_metapaths: int):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.num_metapaths = num_metapaths

        self.W = nn.Parameter(torch.randn(num_metapaths, cin, cout))
        self.bias = nn.Parameter(torch.zeros(num_metapaths, cout))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        _xavier_uniform_(self.W, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, M, cin]  →  [B, M, cout]
        return torch.einsum("bcm,cmn->bcn", x, self.W) + self.bias.unsqueeze(0)


class Transformer(nn.Module):
    """Cross-metapath self-attention (semantic fusion) from SeHGNN."""

    def __init__(self, n_channels: int, num_heads: int = 1, att_drop: float = 0.0, act: str = "none"):
        super().__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert n_channels % (num_heads * 4) == 0, (
            f"n_channels ({n_channels}) must be divisible by num_heads*4 ({num_heads*4})"
        )

        self.query = nn.Linear(n_channels, n_channels // 4)
        self.key   = nn.Linear(n_channels, n_channels // 4)
        self.value = nn.Linear(n_channels, n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.0]))
        self.att_drop = nn.Dropout(att_drop)

        if act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "leaky_relu":
            self.act = nn.LeakyReLU(0.2)
        elif act == "none":
            self.act = lambda x: x
        else:
            raise ValueError(f"Unrecognised activation '{act}' for Transformer")

        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.query, self.key, self.value]:
            m.reset_parameters()
        nn.init.zeros_(self.gamma)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, M, C = x.size()   # batch, num_metapaths, channels
        H = self.num_heads

        f = self.query(x).view(B, M, H, -1).permute(0, 2, 1, 3)   # [B, H, M, C//4H]
        g = self.key(x).view(B, M, H, -1).permute(0, 2, 3, 1)     # [B, H, C//4H, M]
        h = self.value(x).view(B, M, H, -1).permute(0, 2, 1, 3)   # [B, H, M, C//H]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1)  # [B, H, M, M]
        beta = self.att_drop(beta)
        if mask is not None:
            assert mask.size() == torch.Size((B, M))
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h)                                 # [B, H, M, C//H]
        return o.permute(0, 2, 1, 3).reshape(B, M, C) + x          # residual


# ---------------------------------------------------------------------------
# PaperTower
# ---------------------------------------------------------------------------

class PaperTower(nn.Module):
    """
    Encodes a batch of papers (identified by their pre-propagated metapath
    feature vectors) into a single L2-normalised embedding vector.

    Parameters
    ----------
    feat_keys : list[str]
        Ordered list of metapath feature keys, e.g. ["P", "PP", "PCP"].
        The order must match the order of tensors in feat_dict at forward time.
    nfeat : int
        Input feature dimension (768 for SciBERT).
    hidden : int
        Hidden dimension inside the Transformer and projection layers.
    embed_dim : int
        Final embedding dimension (shared with ContextTower).
    n_fp_layers : int
        Number of LinearPerMetapath MLP layers (≥1).
    dropout : float
        Dropout on activations.
    input_drop : float
        Dropout applied to input features before projection.
    att_drop : float
        Attention dropout inside the Transformer.
    num_heads : int
        Number of attention heads in the Transformer.
    act : str
        Activation inside Transformer attention ('none' | 'relu' | 'leaky_relu' | 'sigmoid').
    residual : bool
        If True, add a skip connection from mean(inputs) → hidden before projection head.
    """

    def __init__(
        self,
        feat_keys: list,
        nfeat: int = 768,
        hidden: int = 512,
        embed_dim: int = 256,
        n_fp_layers: int = 2,
        dropout: float = 0.5,
        input_drop: float = 0.1,
        att_drop: float = 0.0,
        num_heads: int = 1,
        act: str = "none",
        residual: bool = False,
    ):
        super().__init__()

        self.feat_keys = sorted(feat_keys)   # canonical order
        self.num_channels = M = len(self.feat_keys)
        self.residual = residual

        self.input_drop = nn.Dropout(input_drop)

        # --- Feature projection: M independent MLPs (LinearPerMetapath) ---
        assert n_fp_layers >= 1, "n_fp_layers must be >= 1"
        layers = [
            LinearPerMetapath(nfeat, hidden, M),
            nn.LayerNorm([M, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_fp_layers - 1):
            layers += [
                LinearPerMetapath(hidden, hidden, M),
                nn.LayerNorm([M, hidden]),
                nn.PReLU(),
                nn.Dropout(dropout),
            ]
        self.feature_projection = nn.Sequential(*layers)

        # --- Cross-metapath semantic fusion ---
        self.semantic_fusion = Transformer(hidden, num_heads=num_heads, att_drop=att_drop, act=act)

        # --- Flatten + project fused metapath vectors ---
        self.fc_after_concat = nn.Linear(M * hidden, hidden)

        # --- Optional residual from raw input ---
        if residual:
            self.res_fc = nn.Linear(nfeat, hidden, bias=False)

        # --- Projection head → shared embedding space ---
        self.proj_head = nn.Linear(hidden, embed_dim)

        self.reset_parameters()

    # ------------------------------------------------------------------
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        for module in self.feature_projection:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.semantic_fusion.reset_parameters()

        nn.init.xavier_uniform_(self.fc_after_concat.weight, gain=gain)
        nn.init.zeros_(self.fc_after_concat.bias)

        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)

        nn.init.xavier_uniform_(self.proj_head.weight, gain=gain)
        nn.init.zeros_(self.proj_head.bias)

    # ------------------------------------------------------------------
    def forward(self, feat_dict: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        feat_dict : dict[str -> Tensor[B, nfeat]]
            Pre-propagated metapath features for a batch of B papers.
            Keys must be a superset of self.feat_keys.

        Returns
        -------
        Tensor[B, embed_dim], L2-normalised.
        """
        # Stack in canonical key order → [B, M, nfeat]
        x = torch.stack([feat_dict[k] for k in self.feat_keys], dim=1)
        x = self.input_drop(x)

        # Optionally stash mean input for residual
        if self.residual:
            x_mean = x.mean(dim=1)   # [B, nfeat]

        # Per-metapath MLP projection → [B, M, hidden]
        x = self.feature_projection(x)

        # Cross-metapath attention → [B, M, hidden]
        x = self.semantic_fusion(x)

        # Flatten + project → [B, hidden]
        B = x.size(0)
        x = self.fc_after_concat(x.reshape(B, -1))

        if self.residual:
            x = x + self.res_fc(x_mean)

        # Projection head → [B, embed_dim]
        x = self.proj_head(x)

        # L2 normalise → unit sphere
        return F.normalize(x, p=2, dim=-1)

if __name__ == "__main__":
    import torch

    feat_keys = ["P", "PP", "PCP"]
    B = 4          # batch of 4 papers
    nfeat = 768
    embed_dim = 256

    model = PaperTower(
        feat_keys=feat_keys,
        nfeat=nfeat,
        hidden=512,
        embed_dim=embed_dim,
        n_fp_layers=2,
    )
    print(model)
    print(f"\nTotal params: {sum(p.numel() for p in model.parameters()):,}")

    # Fake metapath features (normally loaded from .pt files)
    feat_dict = {k: torch.randn(B, nfeat) for k in feat_keys}

    out = model(feat_dict)
    print(f"\nInput:  {B} papers, {len(feat_keys)} metapaths, each {nfeat}-dim")
    print(f"Output: {out.shape}  (expected [{B}, {embed_dim}])")

    # Check L2 normalisation
    norms = out.norm(dim=-1)
    print(f"Output norms: {norms.tolist()}  (expected all ≈ 1.0)")