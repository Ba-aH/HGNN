"""
Count how many citation-context rows fall into train/val/test, based on
which paper each context's citing paper belongs to according to a frozen
split_uris.json (produced by data_split.py / freeze_split_simple.py).

split_uris.json only partitions PAPERS, not contexts -- this script maps
that paper-level split down to the context-level data by looking up each
context's 'citing_uri' field against the train/val/test URI sets.

Reads:
  all_contexts.json  -- list of records like:
    {"context": "...", "cited_uri": "https://citekg.org/resource/paper/...",
     "citing_uri": "https://citekg.org/resource/paper/...", "citing_idx": 1}
  split_uris.json     -- {"train": [uri, ...], "val": [...], "test": [...], "meta": {...}}

Usage:
    python count_contexts_per_split.py  --split split_uris.json --contexts all_contexts.json
"""

import argparse
import json
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(description="Count contexts per train/val/test split.")
    p.add_argument("--split", required=True, help="Path to split_uris.json")
    p.add_argument("--contexts", required=True, help="Path to all_contexts.json")
    p.add_argument("--key", default="citing_uri",
                    help="Field in each context record to match against the "
                         "split's paper URIs (default: citing_uri).")
    return p.parse_args()


def build_lookup(split_data):
    """Returns dict: uri -> split_name ('train'/'val'/'test')."""
    lookup = {}
    for split_name in ("train", "val", "test"):
        for uri in split_data.get(split_name, []):
            lookup[uri] = split_name
    return lookup


def main():
    args = parse_args()

    with open(args.split, encoding="utf-8") as f:
        split_data = json.load(f)
    lookup = build_lookup(split_data)
    print(f"Loaded split: train={len(split_data.get('train', []))}, "
          f"val={len(split_data.get('val', []))}, test={len(split_data.get('test', []))} papers")

    with open(args.contexts, encoding="utf-8") as f:
        contexts = json.load(f)
    print(f"Loaded {len(contexts)} context records from {args.contexts}")

    counts = Counter()
    unmatched_examples = []
    distinct_papers = {"train": set(), "val": set(), "test": set(), "no match": set()}

    for record in contexts:
        uri = record.get(args.key)
        split_name = lookup.get(uri)
        if split_name is None:
            counts["no match"] += 1
            distinct_papers["no match"].add(uri)
            if len(unmatched_examples) < 5:
                unmatched_examples.append(uri)
        else:
            counts[split_name] += 1
            distinct_papers[split_name].add(uri)

    total = len(contexts)
    print(f"\nContext rows: {total}")
    for split_name in ("train", "val", "test", "no match"):
        n = counts.get(split_name, 0)
        pct = 100 * n / total if total else 0
        n_papers = len(distinct_papers[split_name])
        print(f"  {split_name:<10}: {n:>7} contexts ({pct:5.1f}%)  |  {n_papers:>6} distinct papers")

    total_distinct = len(set().union(*distinct_papers.values()))
    print(f"\nTotal distinct '{args.key}' papers seen across all contexts: {total_distinct}")

    if counts.get("no match", 0) > 0:
        print(f"\nSample unmatched '{args.key}' values (paper not found in split_uris.json):")
        for u in unmatched_examples:
            print(f"  {u}")


if __name__ == "__main__":
    main()