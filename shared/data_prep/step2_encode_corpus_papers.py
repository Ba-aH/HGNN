"""
step2_encode_corpus_papers.py
─────────────────────────────
Reads:
  abstracts.json   {paper_uri: abstract_text}
  node_index.json  (paper_id map)
  corpus_ids.pt    (which int IDs are corpus papers)
  paper_uris.json  (int_id → URI lookup)

SciBERT-encodes each abstract (CLS token) in batches.
Papers with no abstract entry get a zero vector and are flagged.

Saves:
  feat_corpus_papers.pt        FloatTensor [N_corpus, 768]
  corpus_missing_abstract.json list of URIs with no abstract
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

ABSTRACTS_FILE = "abstracts.json"
OUT_DIR        = Path(".")
MODEL_NAME     = "allenai/scibert_scivocab_uncased"
BATCH_SIZE     = 32
MAX_LENGTH     = 512
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load index structures ─────────────────────────────────────────────────────
print("Loading index …")
with open(OUT_DIR / "node_index.json") as f:
    node_index = json.load(f)
paper_id: dict[str, int] = node_index["paper"]

with open(OUT_DIR / "paper_uris.json") as f:
    paper_uri_list: list[str] = json.load(f)   # index → URI

corpus_ids: torch.Tensor = torch.load(OUT_DIR / "corpus_ids.pt")
corpus_int_ids = corpus_ids.tolist()
corpus_uris    = [paper_uri_list[i] for i in corpus_int_ids]
N_corpus       = len(corpus_uris)
print(f"  Corpus papers : {N_corpus}")

# ── Load abstracts ────────────────────────────────────────────────────────────
print("Loading abstracts …")
with open(ABSTRACTS_FILE) as f:
    abstracts: dict[str, str] = json.load(f)
print(f"  Abstract entries: {len(abstracts):,}")

# ── Load SciBERT ──────────────────────────────────────────────────────────────
print(f"Loading SciBERT on {DEVICE} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

# ── Encode in batches ─────────────────────────────────────────────────────────
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
    # CLS token is position 0
    return out.last_hidden_state[:, 0, :].cpu()

feat = torch.zeros(N_corpus, 768, dtype=torch.float)
missing: list[str] = []

print("Encoding …")
for batch_start in range(0, N_corpus, BATCH_SIZE):
    batch_uris   = corpus_uris[batch_start : batch_start + BATCH_SIZE]
    batch_texts  = []
    batch_local_idx = []   # positions within this batch that have real text

    for local_i, uri in enumerate(batch_uris):
        text = abstracts.get(uri, "").strip()
        if text:
            batch_texts.append(text)
            batch_local_idx.append(local_i)
        else:
            missing.append(uri)

    if batch_texts:
        vecs = encode_texts(batch_texts)   # [len(batch_texts), 768]
        for vec_i, local_i in enumerate(batch_local_idx):
            global_i = batch_start + local_i
            feat[global_i] = vecs[vec_i]

    done = min(batch_start + BATCH_SIZE, N_corpus)
    print(f"  {done}/{N_corpus}", end="\r")

print()
print(f"  Missing abstracts: {len(missing)}")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(feat, OUT_DIR / "feat_corpus_papers.pt")
with open(OUT_DIR / "corpus_missing_abstract.json", "w") as f:
    f.write('[\n')
    for i, uri in enumerate(missing):
        comma = "," if i < len(missing) - 1 else ""
        f.write(f'  {json.dumps(uri)}{comma}\n')
    f.write(']\n')

print(f"Saved feat_corpus_papers.pt  shape={tuple(feat.shape)}")