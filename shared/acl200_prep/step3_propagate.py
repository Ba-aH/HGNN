"""
step3_propagate.py
────────────────────

Reads:
  feat_P.pt                 [N_total, 768]   (produced by step2_encode_all_papers.py)
  node_index.json
  adj_PP.pt                 [N_papers, N_papers]  raw (unweighted)
  adj_CP_cited.pt            [N_citations, N_papers]
  all_contexts.json          (citing_uri per citation node — for the train mask)
  split_uris.json            (frozen train/val/test citing_uri lists — LEAKAGE GUARD)

LEAKAGE FIX: adj_PP is structural (cito:cites — a paper's own reference
list) and needs no masking; it's static metadata unaffected by val/test
citing activity. adj_CP_cited, however, is built from citation NODES —
i.e. from what citing papers wrote — and feat_PCP is computed as
adj_CP_cited.T @ adj_CP_cited (co-citation via shared contexts). Using the
FULL adj_CP_cited here would mean feat_PCP for every paper (including
train papers) is propagated using co-citation evidence that includes
val/test citing behavior — the same leak shape as feat_PCCon in step4b,
just one hop earlier. Fix: mask adj_CP_cited down to citation nodes whose
citing paper is in the FROZEN TRAIN split BEFORE computing adj_PCP.

Produces:
  feat_PP.pt    [N_total, 768]  1-hop citation-neighbour mean (unmasked — structural)
  feat_PCP.pt   [N_total, 768]  co-citation via context mean (TRAIN-ONLY contexts)

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

# ── Load frozen split (leakage guard) ─────────────────────────────────────────
print("Loading split_uris.json …")
with open(OUT_DIR / "split_uris.json") as f:
    split = json.load(f)
train_citing_uris: set[str] = set(split["train"])
print(f"  Train citing_uris: {len(train_citing_uris):,} "
      f"(val={len(split['val']):,}, test={len(split['test']):,})")

citation_id: dict[str, int] = node_index["citation"]

# ── Build the set of TRAIN citation-node integer IDs ──────────────────────────
# Same construction as step4a_encode_citations.py: citing_uri + citing_idx
# from all_contexts.json → citation node URI → citation_id lookup.
print("Loading all_contexts.json …")
with open("all_contexts.json") as f:
    all_contexts = json.load(f)

train_citation_int_ids: set[int] = set()
for entry in all_contexts:
    citing_uri = entry.get("citing_uri", "").strip()
    citing_idx = entry.get("citing_idx")
    if not citing_uri or citing_idx is None:
        continue
    if citing_uri not in train_citing_uris:
        continue
    cit_uri = citing_uri.replace("/paper/", "/citation/") + f"/{citing_idx}"
    if cit_uri in citation_id:
        train_citation_int_ids.add(citation_id[cit_uri])

print(f"  Train citation-node IDs: {len(train_citation_int_ids):,} / {len(citation_id):,}")

# ── Mask adj_CP_cited down to TRAIN citation-node rows only ──────────────────
# This is what fixes feat_PCP's leak: co-citation counts (adj_CP_cited.T @
# adj_CP_cited) must only reflect train citing-paper contexts, not val/test.
print("Masking adj_CP_cited to train-only citation nodes …")
_indices = adj_CP_cited.indices()   # [2, nnz]
_values  = adj_CP_cited.values()    # [nnz]
_row_ids = _indices[0]

train_id_tensor = torch.tensor(sorted(train_citation_int_ids), dtype=torch.long)
_keep_mask = torch.isin(_row_ids, train_id_tensor)

adj_CP_cited_train = torch.sparse_coo_tensor(
    indices=_indices[:, _keep_mask],
    values=_values[_keep_mask],
    size=adj_CP_cited.shape,
).coalesce()

print(f"  adj_CP_cited        nnz={adj_CP_cited._nnz():,}")
print(f"  adj_CP_cited_train  nnz={adj_CP_cited_train._nnz():,}  "
      f"(dropped {adj_CP_cited._nnz() - adj_CP_cited_train._nnz():,} val/test edges)")

# All downstream use of adj_CP_cited (adj_PCP construction below) now uses
# the train-masked version. adj_PP is left untouched — it's structural.
adj_CP_cited = adj_CP_cited_train

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