"""
freeze_split_simple.py
────────────────────────
Simplest possible split: shuffle paper_uris.json and cut it into
train/val/test by ratio. No filtering, no context/citation logic —
just a deterministic partition of the paper URI list itself.

Reads:
  paper_uris.json   (list of paper URIs, produced by step1_build_graph.py)

Produces:
  split_uris.json
  {
    "train": [paper_uri, ...],
    "val":   [paper_uri, ...],
    "test":  [paper_uri, ...],
    "meta": {...}
  }

Usage:
  python freeze_split_simple.py \
      --paper_uris paper_uris.json \
      --out split_uris.json \
      --seed 42 --train_ratio 0.8 --val_ratio 0.1
"""

import os
import json
import random
import argparse
from datetime import datetime


def compute_split(paper_uris_path, seed, train_ratio, val_ratio):
    print(f"Loading paper URIs from {paper_uris_path} …")
    with open(paper_uris_path, encoding="utf-8") as f:
        paper_uris = json.load(f)
    print(f"  {len(paper_uris):,} paper URIs")

    uris = list(paper_uris)
    rng = random.Random(seed)
    rng.shuffle(uris)

    n_total = len(uris)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)

    train_uris = uris[:n_train]
    val_uris   = uris[n_train : n_train + n_val]
    test_uris  = uris[n_train + n_val :]

    print(f"  Split → train {len(train_uris):,} / "
          f"val {len(val_uris):,} / test {len(test_uris):,}")

    meta = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "n_total": n_total,
        "paper_uris_path": os.path.abspath(paper_uris_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    return {"train": train_uris, "val": val_uris, "test": test_uris, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_uris", required=True)
    ap.add_argument("--out",         default="split_uris.json")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio",   type=float, default=0.1)
    ap.add_argument("--force", action="store_true",
                     help="Overwrite --out if it already exists.")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"[freeze_split_simple] {args.out} already exists. Pass --force to overwrite."
        )

    result = compute_split(args.paper_uris, args.seed, args.train_ratio, args.val_ratio)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote frozen split → {args.out}")
    print(f"  train/val/test paper_uris: "
          f"{len(result['train']):,}/{len(result['val']):,}/{len(result['test']):,}")


if __name__ == "__main__":
    main()
"""
freeze_split_simple.py
────────────────────────
Simplest possible split: shuffle paper_uris.json and cut it into
train/val/test by ratio. No filtering, no context/citation logic —
just a deterministic partition of the paper URI list itself.

Reads:
  paper_uris.json   (list of paper URIs, produced by step1_build_graph.py)

Produces:
  split_uris.json
  {
    "train": [paper_uri, ...],
    "val":   [paper_uri, ...],
    "test":  [paper_uri, ...],
    "meta": {...}
  }

Usage:
  python freeze_split_simple.py \
      --paper_uris paper_uris.json \
      --out split_uris.json \
      --seed 42 --train_ratio 0.8 --val_ratio 0.1
"""

import os
import json
import random
import argparse
from datetime import datetime


def compute_split(paper_uris_path, seed, train_ratio, val_ratio):
    print(f"Loading paper URIs from {paper_uris_path} …")
    with open(paper_uris_path, encoding="utf-8") as f:
        paper_uris = json.load(f)
    print(f"  {len(paper_uris):,} paper URIs")

    uris = list(paper_uris)
    rng = random.Random(seed)
    rng.shuffle(uris)

    n_total = len(uris)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)

    train_uris = uris[:n_train]
    val_uris   = uris[n_train : n_train + n_val]
    test_uris  = uris[n_train + n_val :]

    print(f"  Split → train {len(train_uris):,} / "
          f"val {len(val_uris):,} / test {len(test_uris):,}")

    meta = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "n_total": n_total,
        "paper_uris_path": os.path.abspath(paper_uris_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    return {"train": train_uris, "val": val_uris, "test": test_uris, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_uris", required=True)
    ap.add_argument("--out",         default="split_uris.json")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio",   type=float, default=0.1)
    ap.add_argument("--force", action="store_true",
                     help="Overwrite --out if it already exists.")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"[freeze_split_simple] {args.out} already exists. Pass --force to overwrite."
        )

    result = compute_split(args.paper_uris, args.seed, args.train_ratio, args.val_ratio)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote frozen split → {args.out}")
    print(f"  train/val/test paper_uris: "
          f"{len(result['train']):,}/{len(result['val']):,}/{len(result['test']):,}")


if __name__ == "__main__":
    main()
