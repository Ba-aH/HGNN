"""
step5_encode_PCCon.py
─────────────────────
Reads:
  all_contexts.json    [{context, cited_uri, citing_uri}, ...]
  node_index.json
  paper_uris.json

For each paper P[i], collects all passages where P[i] was the cited paper,
then mean-pools their SciBERT CLS embeddings.

feat_PCCon[i] = mean({ SciBERT(ctx) | ctx.cited_uri == paper_uri[i] })
Papers never cited → zero row.

Saves:
  feat_PCCon.pt               FloatTensor [N_total, 768]
  PCCon_context_counts.json   {uri: n_contexts}  — for diagnostics
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

OUT_DIR    = Path(".")
MODEL_NAME = "allenai/scibert_scivocab_uncased"
BATCH_SIZE = 64
MAX_LENGTH = 256        # contexts are shorter than abstracts; 256 is enough
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load index ────────────────────────────────────────────────────────────────
print("Loading index …")
with open(OUT_DIR / "node_index.json") as f:
    node_index = json.load(f)
paper_id: dict[str, int] = node_index["paper"]
N_total = len(paper_id)

with open(OUT_DIR / "paper_uris.json") as f:
    paper_uri_list: list[str] = json.load(f)   # int_id → URI

# ── Group contexts by cited paper ─────────────────────────────────────────────
print("Loading all_contexts.json …")
with open("all_contexts.json") as f:
    all_contexts = json.load(f)
print(f"  Total context entries: {len(all_contexts):,}")

# cited_uri → list of passage strings
cited_to_contexts: dict[str, list[str]] = defaultdict(list)
skipped = 0
for entry in all_contexts:
    cited_uri = entry.get("cited_uri", "").strip()
    text      = entry.get("context",   "").strip()
    if not cited_uri or not text:
        skipped += 1
        continue
    if cited_uri not in paper_id:
        skipped += 1
        continue
    cited_to_contexts[cited_uri].append(text)

print(f"  Papers with ≥1 context : {len(cited_to_contexts):,}")
print(f"  Skipped entries        : {skipped:,}")

# ── Load SciBERT ──────────────────────────────────────────────────────────────
print(f"Loading SciBERT on {DEVICE} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

def encode_texts(texts: list[str]) -> torch.Tensor:
    """Returns [len(texts), 768] CLS embeddings."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        out = model(**encoded)
    return out.last_hidden_state[:, 0, :].cpu()

# ── Encode: bulk pass over all unique passages, then aggregate per paper ──────
#
# Strategy: flatten all passages into one ordered list, encode in batches,
# then scatter-mean back into per-paper tensors.
# This avoids reloading the model per paper and maximises GPU utilisation.
#
print("Flattening passages for bulk encoding …")

# Build two parallel lists: passage text and its target paper int_id
all_texts:   list[str] = []
all_targets: list[int] = []   # paper int_id for each passage

for uri, passages in cited_to_contexts.items():
    pid = paper_id[uri]
    for p in passages:
        all_texts.append(p)
        all_targets.append(pid)

N_passages = len(all_texts)
print(f"  Total passages to encode: {N_passages:,}")

# ── Bulk encode ───────────────────────────────────────────────────────────────
print("Encoding …")
all_vecs = torch.zeros(N_passages, 768, dtype=torch.float)

for b_start in range(0, N_passages, BATCH_SIZE):
    b_end  = min(b_start + BATCH_SIZE, N_passages)
    batch  = all_texts[b_start:b_end]
    vecs   = encode_texts(batch)          # [batch_size, 768]
    all_vecs[b_start:b_end] = vecs
    if (b_start // BATCH_SIZE) % 20 == 0:
        print(f"  {b_end}/{N_passages}", end="\r")

print(f"\n  Encoding complete.")

# ── Scatter-mean: accumulate sum and count per paper ─────────────────────────
print("Aggregating per paper …")
targets = torch.tensor(all_targets, dtype=torch.long)   # [N_passages]

feat_sum   = torch.zeros(N_total, 768, dtype=torch.float)
feat_count = torch.zeros(N_total,      dtype=torch.float)

feat_sum.index_add_(0, targets, all_vecs)
feat_count.index_add_(0, targets, torch.ones(N_passages, dtype=torch.float))

# Mean-pool: divide only where count > 0
nonzero_mask = feat_count > 0                         # [N_total]
feat_PCCon   = torch.zeros(N_total, 768, dtype=torch.float)
feat_PCCon[nonzero_mask] = (
    feat_sum[nonzero_mask] / feat_count[nonzero_mask].unsqueeze(1)
)

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(feat_PCCon, OUT_DIR / "feat_PCCon.pt")

context_counts = {uri: len(v) for uri, v in cited_to_contexts.items()}
with open(OUT_DIR / "PCCon_context_counts.json", "w") as f:
    items = list(context_counts.items())
    f.write('{\n')
    for i, (uri, count) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        f.write(f'  {json.dumps(uri)}: {count}{comma}\n')
    f.write('}\n')

covered = nonzero_mask.sum().item()
print(f"\nSaved feat_PCCon.pt  shape={tuple(feat_PCCon.shape)}")
print(f"  Papers with features : {covered} / {N_total}")
print(f"  Zero rows            : {N_total - covered}")