"""
step3_propagate.py
────────────────────

Reads:
  feat_P.pt                 [N_total, 768]   (produced by step2_encode_all_papers.py)
  node_index.json
  adj_PP.pt                 [N_papers, N_papers]  raw (unweighted)
  adj_CP_cited.pt            [N_citations, N_papers]

Produces:
  feat_PP.pt    [N_total, 768]  1-hop citation-neighbour mean
  feat_PCP.pt   [N_total, 768]  co-citation via context mean

(feat_P.pt itself is untouched/passed through — already produced by step2)
"""

import json
from pathlib import Path

import torch

OUT_DIR = Path(".")

# ── Load index ────────────────────────────────────────────────────────────────
print("Loading index …")
with open(OUT_DIR / "node_index.json") as f:
    node_index = json.load(f)
paper_id: dict[str, int] = node_index["paper"]
N_total = len(paper_id)

# ── Load feat_P directly (already assembled by step2) ────────────────────────
print("Loading feat_P …")
feat_P = torch.load(OUT_DIR / "feat_P.pt")
assert feat_P.shape == (N_total, 768), f"Unexpected shape: {feat_P.shape}"
print(f"  feat_P  shape={tuple(feat_P.shape)}")

# ── Load adjacency matrices ───────────────────────────────────────────────────
print("Loading adjacency matrices …")
adj_PP        = torch.load(OUT_DIR / "adj_PP.pt").coalesce()         # [N, N]
adj_CP_cited  = torch.load(OUT_DIR / "adj_CP_cited.pt").coalesce()   # [C, N]

N_citations = adj_CP_cited.shape[0]
print(f"  adj_PP        {tuple(adj_PP.shape)}  nnz={adj_PP._nnz():,}")
print(f"  adj_CP_cited  {tuple(adj_CP_cited.shape)}  nnz={adj_CP_cited._nnz():,}")

# ── Helper: row-normalise a sparse COO tensor ─────────────────────────────────
def row_normalise(sp: torch.Tensor) -> torch.Tensor:
    """
    Divide each non-zero by the row sum.
    Rows with degree 0 stay all-zero (no-op, avoids division by zero).
    Returns a new sparse COO tensor.
    """
    sp = sp.coalesce()
    indices = sp.indices()   # [2, nnz]
    values  = sp.values()    # [nnz]

    row_sum = torch.zeros(sp.shape[0], dtype=torch.float)
    row_sum.scatter_add_(0, indices[0], values)

    row_sum_safe = row_sum.clamp(min=1e-9)
    new_values = values / row_sum_safe[indices[0]]

    return torch.sparse_coo_tensor(indices, new_values, sp.shape).coalesce()

# ── Build adj_PCP (co-citation via context) ───────────────────────────────────
print("Building adj_PCP …")
try:
    import scipy.sparse as sp_sci
    import numpy as np

    def to_scipy(t: torch.Tensor):
        t = t.coalesce().cpu()
        idx = t.indices().numpy()
        val = t.values().numpy()
        return sp_sci.csr_matrix((val, (idx[0], idx[1])), shape=t.shape)

    A = to_scipy(adj_CP_cited)        # [C, N]  CSR
    PCP_sci = A.T.dot(A)              # [N, N]  CSR  sparse × sparse

    PCP_sci.setdiag(0)
    PCP_sci.eliminate_zeros()

    cx = PCP_sci.tocoo()
    pcp_indices = torch.tensor(np.vstack([cx.row, cx.col]), dtype=torch.long)
    pcp_values  = torch.tensor(cx.data, dtype=torch.float)
    adj_PCP_raw = torch.sparse_coo_tensor(pcp_indices, pcp_values,
                                           (N_total, N_total)).coalesce()
    print(f"  adj_PCP raw nnz={adj_PCP_raw._nnz():,}  (scipy path)")

except ImportError:
    print("  scipy not available — falling back to dense (may be slow for large graphs)")
    A_dense = adj_CP_cited.to_dense()          # [C, N]
    PCP_dense = A_dense.T @ A_dense            # [N, N]
    PCP_dense.fill_diagonal_(0.0)
    adj_PCP_raw = PCP_dense.to_sparse().coalesce()
    print(f"  adj_PCP raw nnz={adj_PCP_raw._nnz():,}  (dense path)")

# ── Row-normalise both adjacencies ────────────────────────────────────────────
print("Row-normalising …")
adj_PP_norm  = row_normalise(adj_PP)
adj_PCP_norm = row_normalise(adj_PCP_raw)

# ── 1-hop propagation: feat_PP = adj_PP_norm @ feat_P ────────────────────────
print("Propagating feat_PP …")
feat_PP = torch.sparse.mm(adj_PP_norm, feat_P)
torch.save(feat_PP, OUT_DIR / "feat_PP.pt")
print(f"  feat_PP  shape={tuple(feat_PP.shape)}")

# ── 1-hop propagation: feat_PCP = adj_PCP_norm @ feat_P ──────────────────────
print("Propagating feat_PCP …")
feat_PCP = torch.sparse.mm(adj_PCP_norm, feat_P)
torch.save(feat_PCP, OUT_DIR / "feat_PCP.pt")
print(f"  feat_PCP shape={tuple(feat_PCP.shape)}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\nAll done. Output files:")
for name in ["feat_P.pt", "feat_PP.pt", "feat_PCP.pt"]:
    p = OUT_DIR / name
    t = torch.load(p)
    nonzero_rows = (t.abs().sum(dim=1) > 0).sum().item()
    print(f"  {name:20s}  shape={tuple(t.shape)}  "
          f"non-zero rows={nonzero_rows}/{t.shape[0]}")
