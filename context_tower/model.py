"""
context_tower/model.py
----------------------
ContextTower: encodes a citing passage (context text) into a fixed-dim
L2-normalised embedding that lives in the same space as PaperTower's output.

Architecture:
    SciBERT (fine-tuned)           — allenai/scibert_scivocab_uncased
    CLS pooling                    — take hidden state of [CLS] token
    Linear(hidden_size, embed_dim) — project into shared embedding space
    LayerNorm(embed_dim)           — stabilise activations
    L2 normalise                   — unit sphere, consistent with PaperTower

Input:
    input_ids      : LongTensor  [B, seq_len]
    attention_mask : LongTensor  [B, seq_len]
    (token_type_ids are not needed for SciBERT single-sequence inputs)

Output:
    Tensor[B, embed_dim], L2-normalised.

Notes:
- SciBERT hidden_size = 768, matching nfeat in PaperTower.
- embed_dim must match the value used in PaperTower (default 256).
- All SciBERT weights are trainable (fine-tuned end-to-end).
- A lower learning rate for SciBERT vs the projection head is recommended
  in train.py (e.g. 1e-5 for SciBERT, 1e-3 for proj layers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ContextTower(nn.Module):
    """
    Parameters
    ----------
    embed_dim : int
        Output embedding dimension. Must match PaperTower's embed_dim.
    scibert_model_name : str
        HuggingFace model identifier for SciBERT.
    dropout : float
        Dropout applied after CLS pooling, before the projection layer.
    """


    def __init__(
        self,
        embed_dim: int,
        scibert_model_name: str,
        dropout: float,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # --- SciBERT encoder (all weights trainable) ---
        self.scibert = AutoModel.from_pretrained(scibert_model_name)

        # --- Projection head ---
        self.dropout = nn.Dropout(dropout)
        self.proj    = nn.Linear(768, embed_dim) # 768 is the hidden size of SciBERT,it's a property of the model, not a hyperparameter.
                                                 # HIDDEN: it's the dimension of the vector that represents each token after passing through the transformer layers.
        self.norm    = nn.LayerNorm(embed_dim)

        self._init_projection()

    # ------------------------------------------------------------------
    def _init_projection(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.proj.weight, gain=gain)
        nn.init.zeros_(self.proj.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids      : LongTensor [B, seq_len]
        attention_mask : LongTensor [B, seq_len]

        Returns
        -------
        Tensor [B, embed_dim], L2-normalised.
        """
        # SciBERT forward — returns (last_hidden_state, pooler_output, ...)
        outputs = self.scibert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS token: first position of last hidden state → [B, 768]
        cls_emb = outputs.last_hidden_state[:, 0, :]

        # Project into shared embedding space
        x = self.dropout(cls_emb)
        x = self.proj(x)       # [B, embed_dim]
        x = self.norm(x)       # LayerNorm

        # L2 normalise → unit sphere
        return F.normalize(x, p=2, dim=-1)

    # ------------------------------------------------------------------
    def get_param_groups(self, lr_scibert: float, lr_head: float) -> list:
        """
        Returns two parameter groups with different learning rates,
        ready to pass directly to an optimiser.

        Usage in train.py:
            optimizer = torch.optim.Adam(
                model.get_param_groups(lr_scibert=1e-5, lr_head=1e-3)
            )
        """
        return [
            {"params": self.scibert.parameters(), "lr": lr_scibert},
            {"params": list(self.proj.parameters())
                     + list(self.norm.parameters()),  "lr": lr_head},
        ]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     from transformers import AutoTokenizer

#     embed_dim = 256
#     model = ContextTower(embed_dim=embed_dim)
#     print(model)

#     scibert_params = sum(p.numel() for p in model.scibert.parameters())
#     head_params    = sum(p.numel() for p in model.proj.parameters()) \
#                    + sum(p.numel() for p in model.norm.parameters())
#     print(f"\nSciBERT params : {scibert_params:,}")
#     print(f"Head params    : {head_params:,}")
#     print(f"Total params   : {scibert_params + head_params:,}")

#     # Tokenise two fake citing passages
#     tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
#     passages = [
#         "Graph neural networks have been widely adopted for node classification [CITATION].",
#         "As shown by [CITATION], attention mechanisms improve heterogeneous graph learning.",
#     ]
#     enc = tokenizer(passages, padding=True, truncation=True,
#                     max_length=128, return_tensors="pt")

#     model.eval()
#     with torch.no_grad():
#         out = model(enc["input_ids"], enc["attention_mask"])

#     print(f"\nInput:  {len(passages)} passages")
#     print(f"Output: {out.shape}  (expected [{len(passages)}, {embed_dim}])")
#     norms = out.norm(dim=-1)
#     print(f"Output norms: {norms.tolist()}  (expected all ≈ 1.0)")

#     # Check param groups
#     groups = model.get_param_groups()
#     print(f"\nParam groups: {len(groups)} groups")
#     print(f"  Group 0 (SciBERT) lr={groups[0]['lr']}")
#     print(f"  Group 1 (head)    lr={groups[1]['lr']}")