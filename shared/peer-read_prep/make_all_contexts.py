"""
make_all_contexts.py
─────────────────────
Reads merged-kg.ttl and reconstructs all_contexts.json by walking citation
nodes via SPARQL:

  ?citing_paper citekg:hasCitationContext ?cit_node .
  ?cit_node cito:hasCitingEntity ?citing_paper .
  ?cit_node cito:hasCitedEntity  ?cited_paper .
  ?cit_node c4o:hasContext       ?context_text .

Citation node URIs follow the pattern:
  https://citekg.org/resource/citation/{citing_paper_local_id}/{citing_idx}
i.e. exactly citing_uri.replace("/paper/", "/citation/") + f"/{citing_idx}",
which is the same reconstruction step5a.py uses in reverse. citing_idx is
recovered here as the trailing path segment of the citation node URI.

Produces:
  all_contexts.json   [{context, cited_uri, citing_uri, citing_idx}, ...]

Entries are only kept if the citation node has a citing entity, a cited
entity, non-empty context text, and a citing_idx that parses as an int.
"""

import json
from pathlib import Path

from rdflib import Graph, Namespace

CITO   = Namespace("http://purl.org/spar/cito/")
C4O    = Namespace("http://purl.org/spar/c4o/")
CITEKG = Namespace("https://citekg.org/ontology/")

CITATION_PREFIX = "https://citekg.org/resource/citation/"

KG_FILE = "merged-kg.ttl"
OUT_DIR = Path(".")

SPARQL_PREFIXES = """
PREFIX cito: <http://purl.org/spar/cito/>
PREFIX c4o: <http://purl.org/spar/c4o/>
PREFIX citekg: <https://citekg.org/ontology/>
"""

# ── Load graph ────────────────────────────────────────────────────────────────
print(f"Loading {KG_FILE} …")
g = Graph()
g.parse(KG_FILE, format="turtle")
print(f"  {len(g):,} triples loaded")

# ── Query each piece separately, then join in Python (mirrors step1's style)──
print("Querying cito:hasCitingEntity …")
q_citing = SPARQL_PREFIXES + """
SELECT ?cit ?paper WHERE { ?cit cito:hasCitingEntity ?paper . }
"""
citing_of: dict[str, str] = {str(c): str(p) for c, p in g.query(q_citing)}

print("Querying cito:hasCitedEntity …")
q_cited = SPARQL_PREFIXES + """
SELECT ?cit ?paper WHERE { ?cit cito:hasCitedEntity ?paper . }
"""
cited_of: dict[str, str] = {str(c): str(p) for c, p in g.query(q_cited)}

print("Querying c4o:hasContext …")
q_context = SPARQL_PREFIXES + """
SELECT ?cit ?text WHERE { ?cit c4o:hasContext ?text . }
"""
context_of: dict[str, str] = {str(c): str(t).strip() for c, t in g.query(q_context)}

print(f"  citing links  : {len(citing_of):,}")
print(f"  cited links   : {len(cited_of):,}")
print(f"  context texts : {len(context_of):,}")

# ── Union of all citation node URIs seen across the three predicates ─────────
all_cit_uris = set(citing_of) | set(cited_of) | set(context_of)
all_cit_uris = {u for u in all_cit_uris if u.startswith(CITATION_PREFIX)}
print(f"  Citation nodes total (union): {len(all_cit_uris):,}")

# ── Assemble entries ──────────────────────────────────────────────────────────
print("Assembling all_contexts.json entries …")
all_contexts: list[dict] = []

skipped_no_citing = 0
skipped_no_cited = 0
skipped_no_context = 0
skipped_bad_idx = 0

for cit_uri in sorted(all_cit_uris):
    citing_uri = citing_of.get(cit_uri)
    if not citing_uri:
        skipped_no_citing += 1
        continue

    cited_uri = cited_of.get(cit_uri)
    if not cited_uri:
        skipped_no_cited += 1
        continue

    text = context_of.get(cit_uri, "").strip()
    if not text:
        skipped_no_context += 1
        continue

    # Recover citing_idx as the trailing path segment of the citation URI,
    # e.g. .../citation/1234/7 → citing_idx = 7
    tail = cit_uri.rsplit("/", 1)[-1]
    try:
        citing_idx = int(tail)
    except ValueError:
        skipped_bad_idx += 1
        continue

    all_contexts.append({
        "context": text,
        "cited_uri": cited_uri,
        "citing_uri": citing_uri,
        "citing_idx": citing_idx,
    })

print(f"  Entries written             : {len(all_contexts):,}")
print(f"  Skipped (no citing entity)  : {skipped_no_citing:,}")
print(f"  Skipped (no cited entity)   : {skipped_no_cited:,}")
print(f"  Skipped (no/empty context)  : {skipped_no_context:,}")
print(f"  Skipped (bad citing_idx)    : {skipped_bad_idx:,}")

# ── Save ──────────────────────────────────────────────────────────────────────
print("Saving all_contexts.json …")
with open(OUT_DIR / "all_contexts.json", "w", encoding="utf-8") as f:
    f.write('[\n')
    for i, entry in enumerate(all_contexts):
        comma = "," if i < len(all_contexts) - 1 else ""
        f.write(f'  {json.dumps(entry)}{comma}\n')
    f.write(']\n')

print(f"Done. all_contexts.json  entries={len(all_contexts):,}")
