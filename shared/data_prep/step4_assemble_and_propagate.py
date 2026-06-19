"""
step4_assemble_and_propagate.py
────────────────────────────────
Reads:
  feat_corpus_papers.pt     [N_corpus, 768]
  feat_external_papers.pt   [N_external, 768]
  node_index.json
  corpus_ids.pt
  external_ids.pt
  adj_PP.pt                 [N_papers, N_papers]  raw (unweighted)
  adj_CP_citing.pt          [N_citations, N_papers]
  adj_CP_cited.pt           [N_citations, N_papers]

Produces:
  feat_P.pt     [N_total, 768]  raw stacked features
  feat_PP.pt    [N_total, 768]  1-hop citation-neighbour mean
  feat_PCP.pt   [N_total, 768]  co-citation via context mean
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

corpus_ids   = torch.load(OUT_DIR / "corpus_ids.pt")    # [N_corpus]  LongTensor
external_ids = torch.load(OUT_DIR / "external_ids.pt")  # [N_external] LongTensor
N_corpus   = len(corpus_ids)
N_external = len(external_ids)

# ── Load partial feature tensors ─────────────────────────────────────────────
print("Loading partial features …")
feat_corpus   = torch.load(OUT_DIR / "feat_corpus_papers.pt")    # [N_corpus, 768]
feat_external = torch.load(OUT_DIR / "feat_external_papers.pt")  # [N_external, 768]

assert feat_corpus.shape   == (N_corpus,   768), f"Unexpected shape: {feat_corpus.shape}"
assert feat_external.shape == (N_external, 768), f"Unexpected shape: {feat_external.shape}"

# ── Assemble full feature matrix ──────────────────────────────────────────────
# corpus_ids and external_ids carry the global int IDs produced in step 1.
# Scatter each block into the right rows of feat_P.
print("Assembling feat_P …")
feat_P = torch.zeros(N_total, 768, dtype=torch.float)
feat_P[corpus_ids]   = feat_corpus
feat_P[external_ids] = feat_external
torch.save(feat_P, OUT_DIR / "feat_P.pt")
print(f"  feat_P  shape={tuple(feat_P.shape)}")

# ── Load adjacency matrices ───────────────────────────────────────────────────
print("Loading adjacency matrices …")
adj_PP        = torch.load(OUT_DIR / "adj_PP.pt").coalesce()         # [N, N]
adj_CP_citing = torch.load(OUT_DIR / "adj_CP_citing.pt").coalesce()  # [C, N]
adj_CP_cited  = torch.load(OUT_DIR / "adj_CP_cited.pt").coalesce()   # [C, N]

N_citations = adj_CP_citing.shape[0]
print(f"  adj_PP        {tuple(adj_PP.shape)}  nnz={adj_PP._nnz():,}")
print(f"  adj_CP_citing {tuple(adj_CP_citing.shape)}  nnz={adj_CP_citing._nnz():,}")
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

    # Compute row degrees
    row_sum = torch.zeros(sp.shape[0], dtype=torch.float)
    row_sum.scatter_add_(0, indices[0], values)

    # Avoid division by zero
    row_sum_safe = row_sum.clamp(min=1e-9)

    # Scale values
    new_values = values / row_sum_safe[indices[0]]

    return torch.sparse_coo_tensor(indices, new_values, sp.shape).coalesce()

# ── Build adj_PCP (co-citation via context) ───────────────────────────────────
#
# Logic:
#   adj_CP_cited[c, p] = 1 if citation node c cites paper p
#   adj_CP_cited.T     => [N, C]   paper p is cited by citation c
#   adj_CP_cited.T @ adj_CP_cited  => [N, N]
#     entry [i, j] = number of citation nodes that cite both i and j
#     (i.e. i and j co-occur in the same citing passage)
#
# We want to build this in sparse format. Doing sparse @ sparse is supported
# in PyTorch via torch.mm when one is converted to dense, or via bmm.
# For large matrices we stay sparse using scipy temporarily.
#
print("Building adj_PCP …")
try:
    import scipy.sparse as sp_sci
    import numpy as np

    def to_scipy(t: torch.Tensor):
        t = t.coalesce().cpu()
        idx = t.indices().numpy()
        val = t.values().numpy()
        return sp_sci.csr_matrix(
            (val, (idx[0], idx[1])), shape=t.shape
        )

    A = to_scipy(adj_CP_cited)        # [C, N]  CSR
    PCP_sci = A.T.dot(A)              # [N, N]  CSR  sparse × sparse

    # Remove diagonal (a paper is trivially co-cited with itself)
    PCP_sci.setdiag(0)
    PCP_sci.eliminate_zeros()

    cx = PCP_sci.tocoo()
    pcp_indices = torch.tensor(np.vstack([cx.row, cx.col]), dtype=torch.long)
    pcp_values  = torch.tensor(cx.data, dtype=torch.float)
    adj_PCP_raw = torch.sparse_coo_tensor(pcp_indices, pcp_values,
                                           (N_total, N_total)).coalesce()
    print(f"  adj_PCP raw nnz={adj_PCP_raw._nnz():,}  (scipy path)")

except ImportError:
    # Fallback: dense (only feasible for small graphs, ~1-2k papers)
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
#
# Each paper's new feature = mean of its cited neighbours' raw features.
# Papers with no outgoing citations (external papers, Category C) → zero row
# in adj_PP_norm → feat_PP row stays zero (handled by row_normalise guard).
#
print("Propagating feat_PP …")
# torch.sparse mm: sparse [N,N] × dense [N,768] → dense [N,768]
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
