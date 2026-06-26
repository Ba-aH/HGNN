"""
context_tower/model.py  (no-projection variant)
------------------------------------------------
ContextTower: SciBERT CLS embedding → L2 normalise.

No dimensionality reduction. Output lives natively in R^768,
matching PaperTower when run with --hidden 768 --embed_dim 768.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ContextTower(nn.Module):
    """
    Parameters
    ----------
    scibert_model_name : str
        HuggingFace model identifier for SciBERT.
    """

    SCIBERT_HIDDEN = 768   # fixed for allenai/scibert_scivocab_uncased

    def __init__(
        self,
        embed_dim: int = 768,          # kept for API compatibility; must equal 768
        scibert_model_name: str = "allenai/scibert_scivocab_uncased",
        dropout: float = 0.0,          # unused; kept so train.py call-site doesn't break
    ):
        super().__init__()

        if embed_dim != self.SCIBERT_HIDDEN:
            raise ValueError(
                f"No-projection ContextTower requires embed_dim == {self.SCIBERT_HIDDEN}, "
                f"got {embed_dim}."
            )

        self.embed_dim = embed_dim
        self.scibert = AutoModel.from_pretrained(scibert_model_name)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.scibert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]   # [B, 768]
        return F.normalize(cls_emb, p=2, dim=-1)

    # ------------------------------------------------------------------
    def get_param_groups(self, lr_scibert: float = 2e-6, lr_head: float = 1e-4) -> list:
        """
        Only one parameter group now (no projection head).
        lr_head is accepted but ignored so train.py's call-site needs no edits.
        """
        return [
            {"params": self.scibert.parameters(), "lr": lr_scibert},
        ]