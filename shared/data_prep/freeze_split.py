"""
shared/data_prep/freeze_split.py
---------------------------------
One-time script: runs the same filter + shuffle-and-split logic used inside
build_datasets(), against whatever all_contexts.json + node_index.json are
CURRENTLY on disk, and dumps the resulting citing_uri lists to a static file:

    shared/data_prep/split_uris.json
    {
        "train": [citing_uri, ...],
        "val":   [citing_uri, ...],
        "test":  [citing_uri, ...],
        "meta": {
            "seed": 42,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "n_raw": <int>,
            "n_kept": <int>,
            "n_skipped": <int>,
            "n_citing_uris": <int>,
            "all_contexts_path": "...",
            "node_index_path": "...",
            "created_at": "..."
        }
    }

Once this file exists, build_datasets() (see updated dataset.py) will load
these three uri lists directly instead of recomputing the split from
(seed, current data files) — so every future run, and every re-eval of an
existing checkpoint, uses the exact same train/val/test partition regardless
of how all_contexts.json / node_index.json change later.

Usage:
    python freeze_split.py \
        --all_contexts all_contexts.json \
        --node_index   node_index.json \
        --out          split_uris.json \
        --seed 42 --train_ratio 0.8 --val_ratio 0.1

If --out already exists, the script refuses to overwrite unless --force is
passed, since the whole point of this file is to freeze the split ONCE.
"""

import os
import json
import random
import argparse
from datetime import datetime


def compute_split(all_contexts_path, node_index_path, seed, train_ratio, val_ratio):
    print(f"Loading node index from {node_index_path} ...")
    with open(node_index_path, encoding="utf-8") as f:
        node_index = json.load(f)
    paper_uri_to_id = node_index["paper"]
    print(f"  {len(paper_uri_to_id):,} paper nodes in KG")

    print(f"Loading contexts from {all_contexts_path} ...")
    with open(all_contexts_path, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  {len(raw):,} raw records")

    citing_uris_kept = set()
    n_kept, n_skipped = 0, 0

    # Same filter as build_datasets() — must stay in lockstep with dataset.py.
    for item in raw:
        cited_uri        = item.get("cited_uri", "")
        context_text     = item.get("context", "").strip()
        citing_uri       = item.get("citing_uri", "")
        citation_type    = item.get("citation_type", "")
        context_group_id = item.get("context_group_id")

        if not context_text:
            n_skipped += 1
            continue
        if not citation_type:
            n_skipped += 1
            continue
        if context_group_id is None:
            n_skipped += 1
            continue
        if cited_uri not in paper_uri_to_id:
            n_skipped += 1
            continue

        n_kept += 1
        citing_uris_kept.add(citing_uri)

    print(f"  {n_kept:,} records kept, {n_skipped:,} skipped")

    citing_uris = list(citing_uris_kept)
    rng = random.Random(seed)
    rng.shuffle(citing_uris)

    n_uris       = len(citing_uris)
    n_train_uris = int(n_uris * train_ratio)
    n_val_uris   = int(n_uris * val_ratio)

    train_uris = citing_uris[:n_train_uris]
    val_uris   = citing_uris[n_train_uris : n_train_uris + n_val_uris]
    test_uris  = citing_uris[n_train_uris + n_val_uris :]

    print(f"  Split (by citing_uri) → train {len(train_uris):,} / "
          f"val {len(val_uris):,} / test {len(test_uris):,}")

    meta = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "n_raw": len(raw),
        "n_kept": n_kept,
        "n_skipped": n_skipped,
        "n_citing_uris": n_uris,
        "all_contexts_path": os.path.abspath(all_contexts_path),
        "node_index_path": os.path.abspath(node_index_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    return {"train": train_uris, "val": val_uris, "test": test_uris, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_contexts", required=True)
    ap.add_argument("--node_index",   required=True)
    ap.add_argument("--out",          default="split_uris.json")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--train_ratio",  type=float, default=0.8)
    ap.add_argument("--val_ratio",    type=float, default=0.1)
    ap.add_argument("--force", action="store_true",
                     help="Overwrite --out if it already exists.")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"[freeze_split] {args.out} already exists. This file is meant to be "
            f"frozen ONCE and reused by every future run — refusing to overwrite. "
            f"Pass --force if you really intend to re-freeze (this will change what "
            f"'test set' means for every future evaluate.py run against old and new "
            f"checkpoints alike)."
        )

    result = compute_split(
        args.all_contexts, args.node_index, args.seed, args.train_ratio, args.val_ratio
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote frozen split → {args.out}")
    print(f"  train/val/test citing_uris: "
          f"{len(result['train']):,}/{len(result['val']):,}/{len(result['test']):,}")


if __name__ == "__main__":
    main()
