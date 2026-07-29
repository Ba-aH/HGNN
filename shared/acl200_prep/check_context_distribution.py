"""
check_context_distribution.py
───────────────────────────────
Checks how citation contexts (all_contexts.json) distribute across the
frozen train/val/test paper split (split_uris.json), grouped by cited_uri
— i.e. for each context record, look at which paper it CITES, and see
which split that cited paper belongs to.

Reads:
  split_uris.json    {"train": [paper_uri, ...], "val": [...], "test": [...], "meta": {...}}
  all_contexts.json  [{context, cited_uri, citing_uri, citing_idx}, ...]

Prints:
  - Raw count of contexts whose cited_uri falls in train / val / test
  - Percentage of total contexts in each split
  - Count/percentage of contexts whose cited_uri isn't in ANY split
    (e.g. external papers never assigned to a split, if paper_uris.json
    included only a subset, or malformed cited_uri values)

Usage:
  python check_context_distribution.py \
      --split split_uris.json \
      --all_contexts all_contexts.json
"""

import json
import argparse
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="split_uris.json")
    ap.add_argument("--all_contexts", default="all_contexts.json")
    args = ap.parse_args()

    print(f"Loading split from {args.split} …")
    with open(args.split, encoding="utf-8") as f:
        split = json.load(f)

    train_set = set(split["train"])
    val_set   = set(split["val"])
    test_set  = set(split["test"])

    print(f"  train paper_uris: {len(train_set):,}")
    print(f"  val   paper_uris: {len(val_set):,}")
    print(f"  test  paper_uris: {len(test_set):,}")

    print(f"\nLoading contexts from {args.all_contexts} …")
    with open(args.all_contexts, encoding="utf-8") as f:
        all_contexts = json.load(f)
    print(f"  {len(all_contexts):,} total context records")

    # ── Classify each context record by which split its cited_uri belongs to ──
    counts = Counter()
    for item in all_contexts:
        cited_uri = item.get("cited_uri", "")

        if cited_uri in train_set:
            counts["train"] += 1
        elif cited_uri in val_set:
            counts["val"] += 1
        elif cited_uri in test_set:
            counts["test"] += 1
        else:
            counts["unassigned"] += 1

    total = sum(counts.values())
    assert total == len(all_contexts), "Sanity check failed: counts don't sum to total records"

    # ── Print distribution ────────────────────────────────────────────────────
    print("\n── Context distribution by cited_uri split membership ──")
    for split_name in ["train", "val", "test", "unassigned"]:
        n = counts.get(split_name, 0)
        pct = (n / total * 100) if total else 0.0
        print(f"  {split_name:12s}: {n:>8,}  ({pct:5.2f}%)")

    print(f"  {'total':12s}: {total:>8,}  (100.00%)")

    if counts.get("unassigned", 0) > 0:
        print(f"\n[NOTE] {counts['unassigned']:,} context records cite a paper "
              f"not present in any split (e.g. cited_uri malformed, or cited "
              f"paper missing from paper_uris.json used to build the split).")


if __name__ == "__main__":
    main()
