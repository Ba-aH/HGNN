"""
evaluate.py
-----------
Loads a saved checkpoint plus its per-experiment config.json and evaluates
the LCR two-tower model on the held-out test set.

Reports: Recall@1, Recall@5, Recall@10, Recall@20, MRR, nDCG@10

Usage:
    python evaluate.py \
        --checkpoint ~/HGNN/configs/P+PP/no_MLP/best_model.pt \
        --config     ~/HGNN/configs/P+PP/no_MLP/experience.json \
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
from collections import defaultdict


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
    cfg = load_config(os.path.expanduser(args.config))  # load the experiment's saved config.json

    # This lets you re-point data_root/gpu/batch_size at eval time without
    # touching the original training config.
    data_root  =  cfg["data_root"]
    gpu        =  cfg.get("gpu", 0)
    batch_size =  cfg["batch_size"]
    max_length = cfg["max_length"]

    ckpt_path = os.path.expanduser(args.checkpoint)
    # Fall back to CPU automatically if no GPU is available (e.g. running locally)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint (weights only — hyperparameters come from cfg, not ckpt) ---
    # Model architecture args (hidden size, num_heads, use_mlp, etc.) always come
    # from cfg, never from the checkpoint itself — the checkpoint only holds trained weights.
    print("Loading checkpoint ...")
    ckpt = torch.load(ckpt_path, map_location="cpu")  # load to CPU first, move to device after building the model
    print(f" Saved at epoch : {ckpt.get('epoch', '?')}")
    val_mrr = ckpt.get("val_mrr")
    # Report the validation MRR that was recorded at save time, if present (sanity check
    # that this is indeed the best/expected checkpoint before running the full test)
    print(f" Val MRR        : {val_mrr:.4f}\n" if val_mrr is not None else " Val MRR        : ?\n")

    # --- Load dataset ---
    # Must match train.py: grouped file (all_contexts.json), not the
    # raw all_contexts.json, since build_datasets() requires context_group_id
    # on every record and filters out anything missing it.
    print("Loading dataset ...")
    datasets = build_datasets(
        all_contexts_path=os.path.join(data_root, "all_contexts.json"),
        node_index_path=os.path.join(data_root, "node_index.json"),
        max_length=max_length,
        seed=cfg.get("seed", 42),   # same seed as training -> same frozen train/val/test split
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,              # no need to shuffle for evaluation — order doesn't affect metrics
        collate_fn=lcr_collate_fn,
        num_workers=4,
        pin_memory=True,            # speeds up CPU->GPU transfer
    )
    print(f" Test samples : {len(datasets['test']):,}\n")

    # --- Load metapath features (only the ones this experiment actually uses) ---
    # feat_keys comes from cfg, so only the exact feature set this experiment was
    # trained with gets loaded (e.g. P+PP, or P+PP+PCCon) — avoids loading unused tensors.
    print("Loading metapath feature tensors ...")
    feat_keys = cfg["feat_keys"]
    all_paper_feats = {}
    for key in feat_keys:
        path = os.path.join(data_root, f"feat_{key}.pt")
        all_paper_feats[key] = torch.load(path, map_location="cpu")
        print(f" feat_{key}: {all_paper_feats[key].shape}")

    # --- Load corpus + external IDs ---
    # corpus_ids: papers that are part of the main dataset
    # external_ids: papers cited by the corpus but not part of it (must still be
    # candidates, otherwise queries citing them get wrongly skipped — this was
    # the bug that caused 83% of validation queries to be skipped before the fix)
    corpus_ids = torch.load(os.path.join(data_root, "corpus_ids.pt"), map_location="cpu")
    external_ids = torch.load(os.path.join(data_root, "external_ids.pt"), map_location="cpu")

    print(f" Corpus size    : {len(corpus_ids):,}")
    print(f" External size  : {len(external_ids):,}")

    # Unified candidate pool
    # Combine both ID sets into the single pool every query is ranked against.
    # .unique() guards against any accidental overlap between corpus and external IDs.
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
        att_drop=cfg["att_drop"], 
        act=cfg["act"], 
        residual=cfg["residual"], 
        use_mlp=cfg["use_mlp"],
    ).to(device)

    context_tower = ContextTower(
        embed_dim=cfg["embed_dim"],
        dropout=cfg["input_drop"],
        scibert_model_name=cfg["scibert_model_name"],
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
    # pos is the position (index) of each paper ID inside the list candidate_ids_list
    # enumerate(candidate_ids_list) = pairs of (position, paper_id) 
    # We build global_to_pos as a fast reverse lookup table (ID → index) because the embedding matrix and
    # similarity scores are indexed by position, while the ground-truth cited_id is given as a paper ID
    # its an efficient way to map id back to its position in the candidate embedding matrix (precomputed tensors)
    global_to_pos = {gid: pos for pos, gid in enumerate(candidate_ids_list)}

    # --- Evaluation ---
    k_values = [1, 5, 10, 20]
    recall_hits = {k: 0.0 for k in k_values}   # float: per-group contributions can be fractional (Option B)
    mrr_sum = 0.0
    ndcg_sum = 0.0
    n_queries = 0        # will count GROUPS (context_group_id units), not raw citation rows
    n_skipped = 0
    all_ranks = []
    group_ranks = defaultdict(list)  # context_group_id -> list of ranks for papers in that group

    print("Evaluating on test set ...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cited_ids = batch["cited_paper_id"].tolist()
            group_ids = batch["context_group_id"].tolist()

            ctx_emb = context_tower(input_ids, attention_mask)
            sims = torch.matmul(ctx_emb, candidate_embs.T)

            # Phase 1: just compute each row's rank and file it under its group.
            # No metric accumulation happens here anymore — that's Phase 2, after
            # every row in the test set has been seen, since rows belonging to the
            # same context_group_id can land in different batches.
            for i, cited_id in enumerate(cited_ids):
                if cited_id not in global_to_pos:
                    n_skipped += 1
                    continue

                pos = global_to_pos[cited_id]
                sim_row = sims[i]
                rank = int((sim_row > sim_row[pos]).sum().item()) + 1

                all_ranks.append(rank)                    # raw rank, still tracked for rank-distribution stats
                group_ranks[group_ids[i]].append(rank)     # bucket this rank under its context_group_id

    # --- Phase 2: aggregate per context_group_id ---
    # Now that every row in the test set has been ranked, group them by
    # context_group_id and treat each group as ONE evaluation unit — a group
    # with N cited papers contributes fractional Recall@K (papers found in
    # top-K / N) and averaged MRR / nDCG, instead of N independent full-weight
    # rows. This stops multi-citation contexts from dominating the average
    # just because they have more cited papers than a single-citation context.
    print("Aggregating metrics per context_group_id ...")
    for group_id, ranks in group_ranks.items():
        n_papers_in_group = len(ranks)

        for k in k_values:
            hits_in_group = sum(1 for r in ranks if r <= k)
            recall_hits[k] += hits_in_group / n_papers_in_group

        mrr_sum += sum(1.0 / r for r in ranks) / n_papers_in_group
        ndcg_sum += sum(1.0 / math.log2(r + 1) for r in ranks) / n_papers_in_group

        n_queries += 1   # one group = one query, regardless of how many papers it cites

    # Rank distribution
    if all_ranks:
        ranks = sorted(all_ranks)
        n = len(ranks)
        
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