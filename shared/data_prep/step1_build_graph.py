"""
step1_build_graph.py
────────────────────
Reads merged.ttl and extracts:
  - All paper URIs → integer IDs 0..N-1
  - Corpus papers  (have cito:cites outgoing)
  - External papers (only cito:isCitedBy, no outgoing cites)
  - adj_PP        : corpus→any paper  (paper cites paper)
  - adj_CP_citing : citation→corpus paper (hasCitingEntity)
  - adj_CP_cited  : citation→any paper   (hasCitedEntity)

Saves:
  node_index.json       {uri: int_id}
  corpus_ids.pt         LongTensor of corpus paper int IDs
  external_ids.pt       LongTensor of external paper int IDs
  adj_PP.pt             sparse COO FloatTensor [N_papers, N_papers]
  adj_CP_citing.pt      sparse COO FloatTensor [N_citations, N_papers]
  adj_CP_cited.pt       sparse COO FloatTensor [N_citations, N_papers]
"""

import json
from pathlib import Path

import torch
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF

# ── Namespaces ────────────────────────────────────────────────────────────────
CITO    = Namespace("http://purl.org/spar/cito/")
C4O     = Namespace("http://purl.org/spar/c4o/")
CITEKG  = Namespace("https://citekg.org/ontology/")
BIBO    = Namespace("http://purl.org/ontology/bibo/")
DCTERMS = Namespace("http://purl.org/dc/terms/")

PAPER_PREFIX    = "https://citekg.org/resource/paper/"
CITATION_PREFIX = "https://citekg.org/resource/citation/"

OUT_DIR = Path(".")

# ── Load graph ────────────────────────────────────────────────────────────────
print("Loading merged-kg.ttl …")
g = Graph()
g.parse("merged-kg.ttl", format="turtle")
print(f"  {len(g):,} triples loaded")

# ── Collect all paper URIs ────────────────────────────────────────────────────
print("Collecting paper URIs …")

# Corpus papers: subjects of cito:cites triples
corpus_uris: set[str] = set()
for s, _, _ in g.triples((None, CITO.cites, None)):
    uri = str(s)
    if uri.startswith(PAPER_PREFIX):
        corpus_uris.add(uri)

# All paper URIs (corpus + external) — gather from both sides of cito:cites
all_paper_uris: set[str] = set(corpus_uris)
for _, _, o in g.triples((None, CITO.cites, None)):
    uri = str(o)
    if uri.startswith(PAPER_PREFIX):
        all_paper_uris.add(uri)

# Also catch external papers that appear only in cito:isCitedBy
for s, _, _ in g.triples((None, CITO.isCitedBy, None)):
    uri = str(s)
    if uri.startswith(PAPER_PREFIX):
        all_paper_uris.add(uri)
for _, _, o in g.triples((None, CITO.isCitedBy, None)):
    uri = str(o)
    if uri.startswith(PAPER_PREFIX):
        all_paper_uris.add(uri)

external_uris: set[str] = all_paper_uris - corpus_uris

print(f"  Corpus papers  : {len(corpus_uris):,}")
print(f"  External papers: {len(external_uris):,}")
print(f"  Total papers   : {len(all_paper_uris):,}")

# ── Collect all citation node URIs ────────────────────────────────────────────
print("Collecting citation node URIs …")
citation_uris: set[str] = set()

for s, _, _ in g.triples((None, CITO.hasCitingEntity, None)):
    uri = str(s)
    if uri.startswith(CITATION_PREFIX):
        citation_uris.add(uri)
for s, _, _ in g.triples((None, CITO.hasCitedEntity, None)):
    uri = str(s)
    if uri.startswith(CITATION_PREFIX):
        citation_uris.add(uri)

print(f"  Citation nodes : {len(citation_uris):,}")

# ── Build index maps ──────────────────────────────────────────────────────────
# Stable ordering: sort URIs so IDs are deterministic across runs
paper_list    = sorted(all_paper_uris)
citation_list = sorted(citation_uris)

paper_id:    dict[str, int] = {uri: i for i, uri in enumerate(paper_list)}
citation_id: dict[str, int] = {uri: i for i, uri in enumerate(citation_list)}

N_papers    = len(paper_list)
N_citations = len(citation_list)

# ── adj_PP : paper → cited paper (corpus→any) ────────────────────────────────
print("Building adj_PP …")
pp_rows, pp_cols = [], []
for s, _, o in g.triples((None, CITO.cites, None)):
    s_uri, o_uri = str(s), str(o)
    if s_uri in paper_id and o_uri in paper_id:
        pp_rows.append(paper_id[s_uri])
        pp_cols.append(paper_id[o_uri])

adj_PP = torch.sparse_coo_tensor(
    indices=torch.tensor([pp_rows, pp_cols], dtype=torch.long),
    values=torch.ones(len(pp_rows), dtype=torch.float),
    size=(N_papers, N_papers),
).coalesce()
print(f"  adj_PP : {adj_PP._nnz():,} edges")

# ── adj_CP_citing : citation → citing paper ───────────────────────────────────
print("Building adj_CP_citing …")
cp_citing_rows, cp_citing_cols = [], []
for s, _, o in g.triples((None, CITO.hasCitingEntity, None)):
    c_uri, p_uri = str(s), str(o)
    if c_uri in citation_id and p_uri in paper_id:
        cp_citing_rows.append(citation_id[c_uri])
        cp_citing_cols.append(paper_id[p_uri])

adj_CP_citing = torch.sparse_coo_tensor(
    indices=torch.tensor([cp_citing_rows, cp_citing_cols], dtype=torch.long),
    values=torch.ones(len(cp_citing_rows), dtype=torch.float),
    size=(N_citations, N_papers),
).coalesce()
print(f"  adj_CP_citing : {adj_CP_citing._nnz():,} edges")

# ── adj_CP_cited : citation → cited paper ────────────────────────────────────
print("Building adj_CP_cited …")
cp_cited_rows, cp_cited_cols = [], []
for s, _, o in g.triples((None, CITO.hasCitedEntity, None)):
    c_uri, p_uri = str(s), str(o)
    if c_uri in citation_id and p_uri in paper_id:
        cp_cited_rows.append(citation_id[c_uri])
        cp_cited_cols.append(paper_id[p_uri])

adj_CP_cited = torch.sparse_coo_tensor(
    indices=torch.tensor([cp_cited_rows, cp_cited_cols], dtype=torch.long),
    values=torch.ones(len(cp_cited_rows), dtype=torch.float),
    size=(N_citations, N_papers),
).coalesce()
print(f"  adj_CP_cited  : {adj_CP_cited._nnz():,} edges")

# ── Save ──────────────────────────────────────────────────────────────────────
print("Saving outputs …")

node_index = {"paper": paper_id, "citation": citation_id}
with open(OUT_DIR / "node_index.json", "w") as f:
    # One section per type, one "uri": id entry per line inside each block
    f.write('{\n')
    for section_i, (section_key, section_map) in enumerate(node_index.items()):
        f.write(f'  {json.dumps(section_key)}: {{\n')
        items = list(section_map.items())
        for entry_i, (uri, idx) in enumerate(items):
            comma = "," if entry_i < len(items) - 1 else ""
            f.write(f'    {json.dumps(uri)}: {idx}{comma}\n')
        section_comma = "," if section_i < len(node_index) - 1 else ""
        f.write(f'  }}{section_comma}\n')
    f.write('}\n')

corpus_ids   = torch.tensor([paper_id[u] for u in sorted(corpus_uris)],   dtype=torch.long)
external_ids = torch.tensor([paper_id[u] for u in sorted(external_uris)], dtype=torch.long)
torch.save(corpus_ids,    OUT_DIR / "corpus_ids.pt")
torch.save(external_ids,  OUT_DIR / "external_ids.pt")
torch.save(adj_PP,        OUT_DIR / "adj_PP.pt")
torch.save(adj_CP_citing, OUT_DIR / "adj_CP_citing.pt")
torch.save(adj_CP_cited,  OUT_DIR / "adj_CP_cited.pt")

# Also save a URI list (ordered by int ID) for downstream lookups
paper_uri_list = paper_list   # already sorted = index-aligned
with open(OUT_DIR / "paper_uris.json", "w") as f:
    # One URI per line inside a JSON array
    f.write('[\n')
    for i, uri in enumerate(paper_uri_list):
        comma = "," if i < len(paper_uri_list) - 1 else ""
        f.write(f'  {json.dumps(uri)}{comma}\n')
    f.write(']\n')

print("Done.")
print(f"  node_index.json    papers={N_papers}, citations={N_citations}")
print(f"  corpus_ids.pt      {len(corpus_ids)} entries")
print(f"  external_ids.pt    {len(external_ids)} entries")
print(f"  adj_PP.pt          {tuple(adj_PP.shape)}")
print(f"  adj_CP_citing.pt   {tuple(adj_CP_citing.shape)}")
print(f"  adj_CP_cited.pt    {tuple(adj_CP_cited.shape)}")