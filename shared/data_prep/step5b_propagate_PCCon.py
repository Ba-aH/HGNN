"""
step5b_propagate_PCCon.py
─────────────────────────
Reads:
  feat_C.pt          [N_citations, 768]
  adj_CP_cited.pt    [N_citations, N_papers]  (citation → cited paper)
  node_index.json    (for N_total)

Propagation (identical pattern to step4):
  adj_CP_cited.T     [N_papers, N_citations]
  row_normalise(adj_CP_cited.T) @ feat_C  →  feat_PCCon [N_papers, 768]

feat_PCCon[i] = mean SciBERT embedding of all contexts that cite paper i

Saves:
  feat_PCCon.pt    FloatTensor [N_total, 768]
"""

import json
from pathlib import Path

import torch

OUT_DIR = Path(".")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading …")
with open(OUT_DIR / "node_index.json") as f:
    node_index = json.load(f)
N_total = len(node_index["paper"])

feat_C     = torch.load(OUT_DIR / "feat_C.pt")               # [N_cit, 768]
adj_CP_cited = torch.load(OUT_DIR / "adj_CP_cited.pt").coalesce()  # [N_cit, N_papers]

print(f"  feat_C        {tuple(feat_C.shape)}")
print(f"  adj_CP_cited  {tuple(adj_CP_cited.shape)}  nnz={adj_CP_cited._nnz():,}")

# ── row_normalise (copied from step4) ────────────────────────────────────────
def row_normalise(sp: torch.Tensor) -> torch.Tensor:
    sp = sp.coalesce()
    indices  = sp.indices()
    values   = sp.values()

    # Compute row degrees
    row_sum  = torch.zeros(sp.shape[0], dtype=torch.float)
    row_sum.scatter_add_(0, indices[0], values)

    # Avoid division by zero
    row_sum_safe = row_sum.clamp(min=1e-9)

    # Scale values
    new_values   = values / row_sum_safe[indices[0]]

    return torch.sparse_coo_tensor(indices, new_values, sp.shape).coalesce()

# ── Transpose adj_CP_cited → [N_papers, N_citations] ─────────────────────────
# Each row i now lists the citation nodes that cited paper i
adj_cited_T = adj_CP_cited.t().coalesce()   # [N_papers, N_citations]
print(f"  adj_CP_cited.T  {tuple(adj_cited_T.shape)}")

adj_cited_T_norm = row_normalise(adj_cited_T) # [N_papers, N_citations]
                                              # feat_PCCon[i] = mean of SciBERT embeddings of all contexts in which paper i was cited.
# ── Propagate ─────────────────────────────────────────────────────────────────
print("Propagating feat_PCCon …")
feat_PCCon = torch.sparse.mm(adj_cited_T_norm, feat_C)   # [N_papers, 768]

torch.save(feat_PCCon, OUT_DIR / "feat_PCCon.pt")
nonzero = (feat_PCCon.abs().sum(1) > 0).sum().item()
print(f"Saved feat_PCCon.pt  shape={tuple(feat_PCCon.shape)}")
print(f"  Non-zero rows: {nonzero} / {N_total}")
