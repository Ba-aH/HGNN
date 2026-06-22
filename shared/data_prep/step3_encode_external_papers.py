"""
step3_encode_external_papers.py
────────────────────────────────
Reads:
  merged.ttl         (for citekg:hasCitationContext → c4o:hasContext)
  node_index.json
  external_ids.pt
  paper_uris.json

For each external paper:
  - Collect all c4o:hasContext passage texts from its
    citekg:hasCitationContext citation nodes
  - If ≥1 context: feature = mean SciBERT(CLS) over passages
  - If 0 contexts (Category C): feature = zeros(768)

Saves:
  feat_external_papers.pt      FloatTensor [N_external, 768]
  external_has_feature.pt      BoolTensor  [N_external]  (False = zero-padded)
  external_context_counts.json {uri: n_contexts}  — useful for diagnostics
"""

import json
from pathlib import Path

import torch
from rdflib import Graph, Namespace
from transformers import AutoTokenizer, AutoModel

CITEKG = Namespace("https://citekg.org/ontology/")
C4O    = Namespace("http://purl.org/spar/c4o/")

OUT_DIR    = Path(".")
MODEL_NAME = "allenai/scibert_scivocab_uncased"
BATCH_SIZE = 32
MAX_LENGTH = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load graph (needed for context passage text) ──────────────────────────────
print("Loading merged.ttl …")
g = Graph()
g.parse("merged-kg.ttl", format="turtle")
print(f"  {len(g):,} triples")

# ── Load index ────────────────────────────────────────────────────────────────
print("Loading index …")
with open(OUT_DIR / "node_index.json") as f:
    node_index = json.load(f)
paper_id: dict[str, int] = node_index["paper"]

with open(OUT_DIR / "paper_uris.json") as f:
    paper_uri_list: list[str] = json.load(f)

external_ids: torch.Tensor = torch.load(OUT_DIR / "external_ids.pt")
external_int_ids = external_ids.tolist()
external_uris    = [paper_uri_list[i] for i in external_int_ids]
N_external       = len(external_uris)
print(f"  External papers: {N_external}")

# ── Build URI → context passages map from the graph ──────────────────────────
# citekg:hasCitationContext on the cited paper points to citation nodes
# each citation node has c4o:hasContext with the passage text
print("Collecting context passages for external papers …")

from rdflib import URIRef

ext_contexts: dict[str, list[str]] = {uri: [] for uri in external_uris}
ext_uri_set = set(external_uris)

for paper_uri_str in external_uris:
    paper_ref = URIRef(paper_uri_str)
    for _, _, cit_node in g.triples((paper_ref, CITEKG.hasCitationContext, None)):
        # Get the passage text from this citation node
        for _, _, ctx_lit in g.triples((cit_node, C4O.hasContext, None)):
            text = str(ctx_lit).strip()
            if text:
                ext_contexts[paper_uri_str].append(text)

context_counts = {uri: len(v) for uri, v in ext_contexts.items()}
n_with_context = sum(1 for v in context_counts.values() if v > 0)
print(f"  External with ≥1 context : {n_with_context}")
print(f"  External with 0 contexts (Category C) : {N_external - n_with_context}")

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

# ── Encode: for each external paper, mean-pool its context passages ───────────
print("Encoding …")
feat        = torch.zeros(N_external, 768, dtype=torch.float)
has_feature = torch.zeros(N_external, dtype=torch.bool)

for ext_idx, uri in enumerate(external_uris):
    passages = ext_contexts[uri]
    if not passages:
        # Category C — zero vector, flag stays False
        continue

    # Encode in sub-batches (a paper can have many contexts)
    all_vecs = []
    for b_start in range(0, len(passages), BATCH_SIZE):
        batch = passages[b_start : b_start + BATCH_SIZE]
        all_vecs.append(encode_texts(batch))

    paper_vecs  = torch.cat(all_vecs, dim=0)  # [n_ctx, 768]
    feat[ext_idx]        = paper_vecs.mean(dim=0)
    has_feature[ext_idx] = True

    if (ext_idx + 1) % 100 == 0:
        print(f"  {ext_idx + 1}/{N_external}", end="\r")

print(f"\n  Encoded {has_feature.sum().item()} / {N_external} external papers")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(feat,        OUT_DIR / "feat_external_papers.pt")
torch.save(has_feature, OUT_DIR / "external_has_feature.pt")
with open(OUT_DIR / "external_context_counts.json", "w") as f:
    items = list(context_counts.items())
    f.write('{\n')
    for i, (uri, count) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        f.write(f'  {json.dumps(uri)}: {count}{comma}\n')
    f.write('}\n')

print(f"Saved feat_external_papers.pt  shape={tuple(feat.shape)}")
print(f"Saved external_has_feature.pt  True={has_feature.sum().item()}")