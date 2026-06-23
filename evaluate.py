"""
evaluate.py
-----------
Loads a saved checkpoint and evaluates the LCR two-tower model on the
held-out test set.

Reports: Recall@1, Recall@5, Recall@10, Recall@20, MRR, nDCG@10

Usage:
    python evaluate.py \
        --checkpoint ~/HGNN/checkpoints/20260622_120325/best_model.pt \
        --data_root  ~/HGNN/shared/data_prep
"""

import os
import sys
import json
import math
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Path setup ---
ROOT = os.path.expanduser("~/HGNN")
sys.path.insert(0, os.path.join(ROOT, "paper_tower"))
sys.path.insert(0, os.path.join(ROOT, "context_tower"))
sys.path.insert(0, os.path.join(ROOT, "shared", "data_prep"))

from paper_tower.model   import PaperTower
from context_tower.model import ContextTower
from dataset import build_datasets, lcr_collate_fn


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LCR Two-Tower Evaluation")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt checkpoint file")
    parser.add_argument("--data_root",  default="~/HGNN/shared/data_prep")
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size_papers", type=int, default=512,
                        help="Batch size for encoding corpus papers")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device    = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    data_root = os.path.expanduser(args.data_root)
    ckpt_path = os.path.expanduser(args.checkpoint)

    print(f"Device     : {device}")
    print(f"Checkpoint : {ckpt_path}\n")

    # --- Load checkpoint ---
    print("Loading checkpoint ...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_args = ckpt.get("args", {})
    print(f"  Saved at epoch : {ckpt.get('epoch', '?')}")
    print(f"  Val MRR        : {ckpt.get('val_mrr', '?'):.4f}\n")

    embed_dim   = saved_args.get("embed_dim",   256)
    hidden      = saved_args.get("hidden",      512)
    n_fp_layers = saved_args.get("n_fp_layers", 2)
    dropout     = saved_args.get("dropout",     0.5)
    input_drop  = saved_args.get("input_drop",  0.1)

    # --- Load dataset (test split only) ---
    print("Loading dataset ...")
    datasets = build_datasets(
        all_contexts_path = os.path.join(data_root, "all_contexts.json"),
        node_index_path   = os.path.join(data_root, "node_index.json"),
        max_length        = args.max_length,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = lcr_collate_fn,
        num_workers = 4,
        pin_memory  = True,
    )
    print(f"  Test samples : {len(datasets['test']):,}\n")

    # --- Load metapath features ---
    print("Loading metapath feature tensors ...")
    feat_keys = ["P", "PP", "PCP"]
    all_paper_feats = {}
    for key in feat_keys:
        path = os.path.join(data_root, f"feat_{key}.pt")
        all_paper_feats[key] = torch.load(path, map_location="cpu")
        print(f"  feat_{key}: {all_paper_feats[key].shape}")

    corpus_ids = torch.load(os.path.join(data_root, "corpus_ids.pt"), map_location="cpu")
    print(f"  Corpus size  : {len(corpus_ids):,}\n")

    # --- Build models and load weights ---
    paper_tower = PaperTower(
        feat_keys   = feat_keys,
        nfeat       = 768,
        hidden      = hidden,
        embed_dim   = embed_dim,
        n_fp_layers = n_fp_layers,
        dropout     = dropout,
        input_drop  = input_drop,
    ).to(device)

    context_tower = ContextTower(
        embed_dim = embed_dim,
        dropout   = input_drop,
    ).to(device)

    paper_tower.load_state_dict(ckpt["paper_tower"])
    context_tower.load_state_dict(ckpt["context_tower"])
    print("Model weights loaded.\n")

    paper_tower.eval()
    context_tower.eval()

    # --- Precompute corpus embeddings ---
    print("Precomputing corpus paper embeddings ...")
    corpus_ids_list = corpus_ids.tolist()
    corpus_embs = []

    with torch.no_grad():
        for start in tqdm(range(0, len(corpus_ids_list), args.batch_size_papers),
                          desc="  Encoding papers"):
            batch_ids = corpus_ids_list[start : start + args.batch_size_papers]
            batch_feats = {k: v[batch_ids].to(device) for k, v in all_paper_feats.items()}
            emb = paper_tower(batch_feats)
            corpus_embs.append(emb.cpu())

    corpus_embs = torch.cat(corpus_embs, dim=0).to(device)   # [N_corpus, embed_dim]
    print(f"  Corpus embeddings: {corpus_embs.shape}\n")

    # Build global_id → corpus position lookup
    global_to_pos = {gid: pos for pos, gid in enumerate(corpus_ids_list)}

    # --- Evaluate on test set ---
    k_values = [1, 5, 10, 20]
    recall_hits = {k: 0 for k in k_values}
    mrr_sum   = 0.0
    ndcg_sum  = 0.0
    n_queries = 0
    n_skipped = 0

    # Store per-query ranks for analysis
    all_ranks = []

    print("Evaluating on test set ...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Queries"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cited_ids      = batch["cited_paper_id"].tolist()

            ctx_emb = context_tower(input_ids, attention_mask)   # [B, embed_dim]
            sims    = torch.matmul(ctx_emb, corpus_embs.T)       # [B, N_corpus]

            for i, cited_id in enumerate(cited_ids):
                if cited_id not in global_to_pos:
                    n_skipped += 1
                    continue

                pos     = global_to_pos[cited_id]
                sim_row = sims[i]
                rank    = int((sim_row > sim_row[pos]).sum().item()) + 1

                all_ranks.append(rank)
                for k in k_values:
                    if rank <= k:
                        recall_hits[k] += 1

                mrr_sum  += 1.0 / rank
                ndcg_sum += 1.0 / math.log2(rank + 1)
                n_queries += 1

    # --- Report ---
    print("\n" + "="*55)
    print("  TEST SET RESULTS")
    print("="*55)
    print(f"  Queries evaluated : {n_queries:,}")
    print(f"  Skipped (external): {n_skipped:,}")
    print(f"  Corpus size       : {len(corpus_ids_list):,}")
    print("-"*55)
    for k in k_values:
        print(f"  Recall@{k:<3}        : {recall_hits[k] / n_queries:.4f}  ({recall_hits[k]:,} / {n_queries:,})")
    print(f"  MRR               : {mrr_sum  / n_queries:.4f}")
    print(f"  nDCG@10           : {ndcg_sum / n_queries:.4f}")
    print("="*55)

    # --- Rank distribution ---
    ranks = sorted(all_ranks)
    n = len(ranks)
    print(f"\n  Rank distribution (n={n:,}):")
    print(f"  Median rank : {ranks[n//2]}")
    print(f"  Mean rank   : {sum(ranks)/n:.1f}")
    print(f"  Rank=1      : {ranks.count(1):,} ({ranks.count(1)/n*100:.1f}%)")
    print(f"  Rank≤5      : {sum(r<=5  for r in ranks):,} ({sum(r<=5  for r in ranks)/n*100:.1f}%)")
    print(f"  Rank≤10     : {sum(r<=10 for r in ranks):,} ({sum(r<=10 for r in ranks)/n*100:.1f}%)")
    print(f"  Rank>100    : {sum(r>100 for r in ranks):,} ({sum(r>100 for r in ranks)/n*100:.1f}%)")

    # --- Save results to JSON ---
    results = {
        "checkpoint": ckpt_path,
        "epoch":      ckpt.get("epoch"),
        "val_mrr":    ckpt.get("val_mrr"),
        "n_queries":  n_queries,
        "n_skipped":  n_skipped,
        "corpus_size": len(corpus_ids_list),
        "Recall@1":   recall_hits[1]  / n_queries,
        "Recall@5":   recall_hits[5]  / n_queries,
        "Recall@10":  recall_hits[10] / n_queries,
        "Recall@20":  recall_hits[20] / n_queries,
        "MRR":        mrr_sum  / n_queries,
        "nDCG@10":    ndcg_sum / n_queries,
        "median_rank": ranks[n//2],
        "mean_rank":   sum(ranks) / n,
    }

    out_path = os.path.join(os.path.dirname(ckpt_path), "test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()