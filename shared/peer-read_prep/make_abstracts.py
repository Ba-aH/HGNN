"""
make_abstracts.py
──────────────────
Reads merged-kg.ttl and extracts each paper's abstract text via SPARQL:

  - ?paper bibo:abstract ?text

Produces:
  abstracts.json   {paper_uri: abstract_text}

Only papers whose URI starts with PAPER_PREFIX are kept, and only
non-empty abstract strings are written (blank/whitespace-only literals
are dropped).
"""

import json
from pathlib import Path

from rdflib import Graph, Namespace

BIBO = Namespace("http://purl.org/ontology/bibo/")

PAPER_PREFIX = "https://citekg.org/resource/paper/"

KG_FILE = "merged-kg.ttl"
OUT_DIR = Path(".")

SPARQL_PREFIXES = """
PREFIX bibo: <http://purl.org/ontology/bibo/>
"""

# ── Load graph ────────────────────────────────────────────────────────────────
print(f"Loading {KG_FILE} …")
g = Graph()
g.parse(KG_FILE, format="turtle")
print(f"  {len(g):,} triples loaded")

# ── Query abstracts ───────────────────────────────────────────────────────────
print("Querying bibo:abstract …")
q_abstract = SPARQL_PREFIXES + """
SELECT ?s ?o WHERE { ?s bibo:abstract ?o . }
"""

abstracts: dict[str, str] = {}
skipped_not_paper = 0
skipped_empty = 0

for s, o in g.query(q_abstract):
    uri = str(s)
    text = str(o).strip()

    if not uri.startswith(PAPER_PREFIX):
        skipped_not_paper += 1
        continue
    if not text:
        skipped_empty += 1
        continue

    abstracts[uri] = text

print(f"  Abstracts found        : {len(abstracts):,}")
print(f"  Skipped (not a paper)  : {skipped_not_paper:,}")
print(f"  Skipped (empty text)   : {skipped_empty:,}")

# ── Save ──────────────────────────────────────────────────────────────────────
print("Saving abstracts.json …")
with open(OUT_DIR / "abstracts.json", "w", encoding="utf-8") as f:
    items = list(abstracts.items())
    f.write('{\n')
    for i, (uri, text) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        f.write(f'  {json.dumps(uri)}: {json.dumps(text)}{comma}\n')
    f.write('}\n')

print(f"Done. abstracts.json  entries={len(abstracts):,}")
