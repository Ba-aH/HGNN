# Shared (`shared/`)

This folder contains the two dataset preparation pipelines that turn a knowledge graph into the metapath features and frozen splits consumed by training. Each dataset (ArgKG and ACL-200) has its own pipeline, but both follow the same overall structure: build a graph → encode papers → propagate/pool features across metapaths → encode citation contexts → freeze the train/val/test split.

The output of these pipelines — feature tensors, adjacency matrices, and split files — is what `data_root` in a training `config.json` should point to.

## `data_prep/` — ArgKG pipeline

Builds the features for the custom ArgKG knowledge graph, which distinguishes **corpus papers** (scraped directly, full text available) from **external papers** (only reachable through citations, no abstract).

| Script | Purpose |
|---|---|
| `step1_build_graph.py` | Loads `merged-kg.ttl` and builds the graph structure (nodes, adjacency) used by later steps. |
| `step2_encode_corpus_papers.py` | Computes `X^P` for corpus papers: SciBERT encoding of title + abstract. |
| `step3_encode_external_papers.py` | Computes `X^P` for external papers, which lack an abstract, by mean-pooling SciBERT encodings of their citation contexts instead. |
| `step4_assemble_and_propagate.py` | Assembles the full feature matrix and propagates `X^PP` (1-hop citation neighbourhood) across the citation adjacency (`adj_PP.pt`). |
| `step5a_encode_citations.py` | Encodes every citation context with SciBERT. |
| `step5b_propagate_PCCon.py` | Pools citation-context encodings per paper to produce `X^PC`. |
| `freeze_split.py` | Fixes the train/val/test split and writes it to `split_uris.json`, so every experiment run trains/evaluates on the exact same split. |
| `check_consistency_all_context.py` | Sanity-checks context coverage and counts against the graph. |
| `group_aware_sampler.py` | Batch sampler that keeps sibling citation contexts (multi-citation markers) out of the same batch. |
| `dataset.py` | PyTorch `Dataset`/loading logic used by the training scripts. |

Key data files (zipped in the repo — unzip before use):

- `merged-kg.ttl` (from `merged-kg.zip`) — the constructed ArgKG knowledge graph in RDF/Turtle.
- `all_contexts.json` (from `all_contexts.zip`) — all extracted citation contexts.
- `abstracts.json` — title/abstract text for corpus papers.
- `adj_PP.pt`, `adj_CP_cited.pt`, `adj_CP_citing.pt` — precomputed adjacency tensors.
- `corpus_ids.pt`, `external_ids.pt` — node id splits by paper type.
- `node_index.json`, `paper_uris.json` — id ↔ URI lookup tables.
- `split_uris.json` — frozen train/val/test split (output of `freeze_split.py`).
- `corpus_missing_abstract.json`, `external_context_counts.json` — bookkeeping/diagnostic outputs.

Run order:

```bash
cd shared/data_prep
unzip merged-kg.zip
unzip all_contexts.zip

python step1_build_graph.py
python step2_encode_corpus_papers.py
python step3_encode_external_papers.py
python step4_assemble_and_propagate.py
python step5a_encode_citations.py
python step5b_propagate_PCCon.py
python freeze_split.py
```

## `acl200_prep/` — ACL-200 pipeline

Builds the same set of features for the ACL-200 benchmark. Unlike ArgKG, every paper in ACL-200 has an abstract, so there is no corpus/external distinction — all papers are treated uniformly.

| Script | Purpose |
|---|---|
| `step1_build_graph.py` | Builds the ACL-200 graph structure from `acl200_with_id.csv`. |
| `step2_encode_papers.py` | Computes `X^P` for every paper (title + abstract, SciBERT). |
| `step3_propagate.py` | Propagates `X^PP` across the citation adjacency. |
| `step4a_encode_citations.py` | Encodes every citation context with SciBERT. |
| `step4b_propagate_PCCon.py` | Pools citation-context encodings per paper to produce `X^PC`. |
| `data_split.py` | Builds the train/val/test split for ACL-200. |
| `make_abstracts.py` | Builds `abstracts.json` from raw ACL-200 data. |
| `make_all_contexts.py` | Builds `all_contexts.json`, reading from the full `citation_context` field and inserting a `[cit]` marker via regex fuzzy matching. |
| `check_context_distribution.py`, `count_contexts_per_split.py` | Diagnostics on how contexts are distributed across papers/splits. |
| `group_aware_sampler.py`, `dataset.py` | Same role as in `data_prep/`. |

Key data files:

- `acl200_with_id.csv` — raw ACL-200 records with assigned ids.
- `abstracts.json` (from `abstracts.zip`) — title/abstract text for all papers.
- `all_contexts.json` (from `all_contexts.zip`) — all extracted citation contexts.
- `node_index.json`, `paper_uris.json`, `split_uris.json` — same role as in `data_prep/`.

Run order:

```bash
cd shared/acl200_prep
unzip abstracts.zip
unzip all_contexts.zip

python step1_build_graph.py
python step2_encode_papers.py
python step3_propagate.py
python step4a_encode_citations.py
python step4b_propagate_PCCon.py
```

## Notes

- Both pipelines must be run in order: later steps depend on files produced by earlier ones (e.g. `step4_assemble_and_propagate.py` needs the outputs of `step2`/`step3`).
- `split_uris.json` should not be regenerated between experiment runs on the same dataset: keeping it fixed is what makes Recall@n/MRR comparable across configurations.
- Once a pipeline has been run, point a training `config.json`'s `data_root` field at the corresponding folder (`shared/data_prep/` or `shared/acl200_prep/`).