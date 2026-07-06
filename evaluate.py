"""
evaluate.py
-----------
Loads a saved checkpoint plus its per-experiment config.json and evaluates
the LCR two-tower model on the held-out test set.

Reports: Recall@1, Recall@5, Recall@10, Recall@20, MRR, nDCG@10

Usage:
    python evaluate.py \
        --checkpoint ~/HGNN/configs/P+PP/MLP_500/exp_PP_MLP_hidden500(run1)/best_model.pt \
        --config     ~/HGNN/configs/P+PP/MLP_500/exp_PP_MLP_hidden500(run1)/config.json \
        --data_root  ~/HGNN/shared/data_prep

--data_root defaults to cfg["data_root"] from the config file if not passed
explicitly, so in the common case you only need --checkpoint and --config.
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

from paper_tower.model import PaperTower
from context_tower.model import ContextTower
from dataset import build_datasets, lcr_collate_fn


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="LCR Two-Tower Evaluation (Full Corpus + External)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt checkpoint file")
    parser.add_argument("--config", required=True,
                        help="Path to the experiment's config.json (same one train.py saved)")
    parser.add_argument("--data_root", default=None,
                        help="Overrides cfg['data_root'] from config.json if provided")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Overrides cfg['gpu'] from config.json if provided")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Overrides cfg['batch_size'] from config.json if provided")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Overrides cfg['max_length'] from config.json if provided")
    parser.add_argument("--batch_size_papers", type=int, default=512,
                        help="Batch size for encoding papers")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(os.path.expanduser(args.config))

    # CLI overrides take priority; otherwise fall back to the experiment's own config.
    data_root  = os.path.expanduser(args.data_root if args.data_root is not None else cfg["data_root"])
    gpu        = args.gpu        if args.gpu        is not None else cfg.get("gpu", 0)
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    max_length = args.max_length if args.max_length is not None else cfg["max_length"]

    ckpt_path = os.path.expanduser(args.checkpoint)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"  Experiment : {cfg.get('experiment_name', '?')}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Config     : {args.config}")
    print(f"  Device     : {device}")
    print("=" * 60 + "\n")

    # --- Load checkpoint (weights only — hyperparameters come from cfg, not ckpt) ---
    print("Loading checkpoint ...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f" Saved at epoch : {ckpt.get('epoch', '?')}")
    val_mrr = ckpt.get("val_mrr")
    print(f" Val MRR        : {val_mrr:.4f}\n" if val_mrr is not None else " Val MRR        : ?\n")

    # --- Load dataset ---
    # Must match train.py: grouped file (all_contexts_grouped.json), not the
    # raw all_contexts.json, since build_datasets() requires context_group_id
    # on every record and filters out anything missing it.
    print("Loading dataset ...")
    datasets = build_datasets(
        all_contexts_path=os.path.join(data_root, "all_contexts_grouped.json"),
        node_index_path=os.path.join(data_root, "node_index.json"),
        max_length=max_length,
        seed=cfg.get("seed", 42),
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lcr_collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    print(f" Test samples : {len(datasets['test']):,}\n")

    # --- Load metapath features (only the ones this experiment actually uses) ---
    print("Loading metapath feature tensors ...")
    feat_keys = cfg["feat_keys"]
    all_paper_feats = {}
    for key in feat_keys:
        path = os.path.join(data_root, f"feat_{key}.pt")
        all_paper_feats[key] = torch.load(path, map_location="cpu")
        print(f" feat_{key}: {all_paper_feats[key].shape}")

    # --- Load corpus + external IDs ---
    corpus_ids = torch.load(os.path.join(data_root, "corpus_ids.pt"), map_location="cpu")
    external_ids = torch.load(os.path.join(data_root, "external_ids.pt"), map_location="cpu")

    print(f" Corpus size    : {len(corpus_ids):,}")
    print(f" External size  : {len(external_ids):,}")

    # Unified candidate pool
    all_candidate_ids = torch.cat([corpus_ids, external_ids]).unique()  # ensure no duplicates
    candidate_ids_list = all_candidate_ids.tolist()
    print(f" Total candidates : {len(candidate_ids_list):,}\n")

    # --- Build models (hyperparameters strictly from this experiment's config.json) ---
    paper_tower = PaperTower(
        feat_keys=feat_keys,
        nfeat=768,
        num_heads=cfg["num_heads"],
        hidden=cfg["hidden"],
        embed_dim=cfg["embed_dim"],
        n_fp_layers=cfg["n_fp_layers"],
        dropout=cfg["dropout"],
        input_drop=cfg["input_drop"],
    ).to(device)

    context_tower = ContextTower(
        embed_dim=cfg["embed_dim"],
        dropout=cfg["input_drop"],
    ).to(device)

    paper_tower.load_state_dict(ckpt["paper_tower"])
    context_tower.load_state_dict(ckpt["context_tower"])

    paper_tower.eval()
    context_tower.eval()

    # --- Precompute embeddings for ALL candidates ---
    print("Precomputing embeddings for ALL papers (corpus + external)...")
    candidate_embs = []

    with torch.no_grad():
        for start in tqdm(range(0, len(candidate_ids_list), args.batch_size_papers),
                          desc="Encoding papers"):
            batch_ids = candidate_ids_list[start:start + args.batch_size_papers]

            batch_feats = {
                k: v[batch_ids].to(device) for k, v in all_paper_feats.items()
            }

            emb = paper_tower(batch_feats)
            candidate_embs.append(emb.cpu())

    candidate_embs = torch.cat(candidate_embs, dim=0).to(device)
    print(f" All candidate embeddings: {candidate_embs.shape}\n")

    # Global ID → position in candidate_embs
    global_to_pos = {gid: pos for pos, gid in enumerate(candidate_ids_list)}

    # --- Evaluation ---
    k_values = [1, 5, 10, 20]
    recall_hits = {k: 0 for k in k_values}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    n_queries = 0
    n_skipped = 0
    all_ranks = []

    print("Evaluating on test set ...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Queries"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cited_ids = batch["cited_paper_id"].tolist()

            ctx_emb = context_tower(input_ids, attention_mask)  # [B, embed_dim]

            # Similarity against ALL candidates
            sims = torch.matmul(ctx_emb, candidate_embs.T)  # [B, N_candidates]

            for i, cited_id in enumerate(cited_ids):
                if cited_id not in global_to_pos:
                    n_skipped += 1
                    continue

                pos = global_to_pos[cited_id]
                sim_row = sims[i]

                # Compute rank
                rank = int((sim_row > sim_row[pos]).sum().item()) + 1

                all_ranks.append(rank)

                for k in k_values:
                    if rank <= k:
                        recall_hits[k] += 1

                mrr_sum += 1.0 / rank
                ndcg_sum += 1.0 / math.log2(rank + 1)
                n_queries += 1

    # --- Results ---
    print("\n" + "="*65)
    print(" FULL TEST SET RESULTS (Corpus + External)")
    print("="*65)
    print(f" Experiment        : {cfg.get('experiment_name', '?')}")
    print(f" Queries evaluated : {n_queries:,}")
    print(f" Skipped           : {n_skipped:,}  (should be near 0)")
    print(f" Total candidates  : {len(candidate_ids_list):,}")
    print("-"*65)

    for k in k_values:
        print(f" Recall@{k:<3} : {recall_hits[k] / n_queries:.4f} "
              f"({recall_hits[k]:,} / {n_queries:,})")

    print(f" MRR       : {mrr_sum / n_queries:.4f}")
    print(f" nDCG@10   : {ndcg_sum / n_queries:.4f}")
    print("="*65)

    # Rank distribution
    if all_ranks:
        ranks = sorted(all_ranks)
        n = len(ranks)
        print(f"\n Rank distribution (n={n:,}):")
        print(f" Median rank : {ranks[n//2]}")
        print(f" Mean rank   : {sum(ranks)/n:.1f}")
        print(f" Rank=1      : {ranks.count(1):,} ({ranks.count(1)/n*100:.1f}%)")
        print(f" Rank≤5      : {sum(r<=5 for r in ranks):,} ({sum(r<=5 for r in ranks)/n*100:.1f}%)")
        print(f" Rank≤10     : {sum(r<=10 for r in ranks):,} ({sum(r<=10 for r in ranks)/n*100:.1f}%)")
        print(f" Rank>100    : {sum(r>100 for r in ranks):,} ({sum(r>100 for r in ranks)/n*100:.1f}%)")

    # Save results — tagged with experiment_name so results from different
    # experiments never collide if you later gather them into one place.
    results = {
        "experiment_name": cfg.get("experiment_name"),
        "checkpoint": ckpt_path,
        "config": args.config,
        "epoch": ckpt.get("epoch"),
        "n_queries": n_queries,
        "n_skipped": n_skipped,
        "total_candidates": len(candidate_ids_list),
        "Recall@1": recall_hits[1] / n_queries,
        "Recall@5": recall_hits[5] / n_queries,
        "Recall@10": recall_hits[10] / n_queries,
        "Recall@20": recall_hits[20] / n_queries,
        "MRR": mrr_sum / n_queries,
        "nDCG@10": ndcg_sum / n_queries,
        "median_rank": ranks[n//2] if all_ranks else None,
        "mean_rank": sum(ranks)/n if all_ranks else None,
    }

    out_path = os.path.join(os.path.dirname(ckpt_path), "test_results_full.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()