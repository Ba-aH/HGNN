"""
train.py
--------
Training script for the LCR two-tower retrieval model.
Reads all hyperparameters from a JSON config file passed via --config.

ADAPTED for the current dataset: no context_group_id / multi-citation
grouping exists (confirmed: adj_PCP has 0 non-zero entries — every
citation node cites exactly one paper). Changes from the original:
  - GroupAwareBatchSampler removed — standard shuffled DataLoader used
    instead, since there are no sibling rows to keep apart.
  - evaluate() no longer aggregates by context_group_id. Every row is
    its own evaluation unit (equivalent to every "group" having size 1),
    so Recall@K/MRR/nDCG collapse to standard per-record metrics.
  - Candidate pool is now ALL papers (no corpus/external split) — loaded
    directly from node_index.json instead of concatenating
    corpus_ids.pt + external_ids.pt.

Architecture:
    ContextTower  — SciBERT (fine-tuned) → [B, embed_dim]
    PaperTower    — SeHGNN metapath fusion → [N, embed_dim]

Loss: InfoNCE over in-batch negatives (unchanged)
    - Anchor   : context embedding
    - Positive : cited paper embedding
    - Negatives: all other papers in the batch

Evaluation metrics: Recall@K (K=1,5,10,20), MRR, nDCG@10
    - Corpus index is a FROZEN SNAPSHOT built before each epoch's training step.

Usage:
    python train.py --config configs/P+PP/no_MLP/experience.json

    nohup python train_baseline.py --config configs/P+PP/MLP_500/experience.json >myoutfile 2>&1 &
"""

import os
import sys
import json
import math
import argparse
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.expanduser("~/HGNN")
sys.path.insert(0, os.path.join(ROOT, "paper_tower"))
sys.path.insert(0, os.path.join(ROOT, "context_tower"))
sys.path.insert(0, os.path.join(ROOT, "shared", "acl200_prep"))

from paper_tower.model   import PaperTower
from context_tower.model import ContextTower
from dataset import build_datasets, lcr_collate_fn
# GroupAwareBatchSampler intentionally NOT imported — no multi-citation
# grouping exists in this dataset (every citation node cites exactly one
# paper; see adj_PCP nnz=0 in step3_propagate.py's log).


# ---------------------------------------------------------------------------
# Config — load JSON, no defaults, every key must be present
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# InfoNCE contrastive loss — plain single-positive, diagonal target
#
# Exactly one correct column per row (the diagonal): row i's only positive
# is paper_emb[i], everything else in the batch is a negative. No group-based
# masking or multi-positive averaging — cited_uri is the sole ground truth
# for its row and nothing is relaxed.
#
# UNCHANGED from the original: this dataset has no sibling rows sharing
# identical context_text (no multi-citation contexts), so the "identical
# context, different true paper" problem the sampler used to guard against
# simply doesn't occur here. Standard shuffled batches are safe as-is.
# ---------------------------------------------------------------------------
def infonce_loss(ctx_emb, paper_emb, temperature):
    # Compute similarity between every context and every paper in the batch → [B, B]
    logits = torch.matmul(ctx_emb, paper_emb.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_c2p = F.cross_entropy(logits,   labels)  # context → paper
    loss_p2c = F.cross_entropy(logits.T, labels)  # paper → context

    return (loss_c2p + loss_p2c) / 2.0


# ---------------------------------------------------------------------------
# Build frozen candidate index
# Runs PaperTower over all papers ONCE before training starts each epoch.
# The resulting matrix is fixed for the entire evaluation of that epoch —
# weights keep updating during training but the index does not.
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_candidate_index(paper_tower, all_paper_feats, candidate_ids, device):
    was_training = paper_tower.training
    paper_tower.eval()

    embs = []
    ids  = candidate_ids.tolist()

    for start in tqdm(range(0, len(ids), 512), desc="  Building index", leave=False):
        batch_ids   = ids[start : start + 512]
        batch_feats = {k: v[batch_ids].to(device) for k, v in all_paper_feats.items()}
        embs.append(paper_tower(batch_feats).cpu())

    paper_tower.train(was_training)

    return torch.cat(embs, dim=0)


# ---------------------------------------------------------------------------
# Evaluation — ranks all candidates for each val/test context
#
# ADAPTED: no context_group_id exists in this dataset, so every row is its
# own evaluation unit (equivalent to every group having size 1 in the
# original multi-citation-aware version). Recall@K/MRR/nDCG here are the
# standard per-record metrics.
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(context_tower, loader, candidate_embs, candidate_ids, device):
    """
    Every row (context, cited_paper_id) is evaluated independently — one
    query per record. Recall@K/MRR/nDCG are standard per-record metrics
    (no multi-citation grouping in this dataset).
    """
    context_tower.eval()

    cand_dev = candidate_embs.to(device)

    # --- Zero-vector filtering (unchanged) --------------------------------
    nonzero_mask = cand_dev.abs().sum(dim=1) != 0
    n_zero_candidates = (~nonzero_mask).sum().item()
    cand_dev = cand_dev[nonzero_mask]

    kept_ids = [gid for gid, keep in zip(candidate_ids.tolist(), nonzero_mask.tolist()) if keep]
    global_to_pos = {gid: pos for pos, gid in enumerate(kept_ids)}

    k_values = [1, 5, 10, 20]

    recall_hits = {k: 0.0 for k in k_values}
    mrr_sum, ndcg_sum, n_queries = 0.0, 0.0, 0
    n_skipped = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cited_ids = batch["cited_paper_id"].tolist()

            ctx_emb = context_tower(input_ids, attention_mask)
            sims = torch.matmul(ctx_emb, cand_dev.T)

            for i, cited_id in enumerate(cited_ids):
                if cited_id not in global_to_pos:
                    n_skipped += 1
                    continue

                pos = global_to_pos[cited_id]
                true_score = sims[i][pos]

                beats_true_paper = sims[i] > true_score
                num_candidates_ahead = int(beats_true_paper.sum().item())
                rank = num_candidates_ahead + 1

                for k in k_values:
                    if rank <= k:
                        recall_hits[k] += 1.0

                mrr_sum  += 1.0 / rank
                ndcg_sum += 1.0 / math.log2(rank + 1)
                n_queries += 1

    metrics = {f"Recall@{k}": recall_hits[k] / n_queries for k in k_values}
    metrics.update({"MRR": mrr_sum / n_queries, "nDCG@10": ndcg_sum / n_queries,
                "n_queries": n_queries, "n_skipped": n_skipped,
                "n_zero_candidates_dropped": n_zero_candidates})
    return metrics


# ---------------------------------------------------------------------------
# One training epoch
# For each batch: encode B contexts + their B cited papers → InfoNCE loss
# → backprop → update both towers.
# Returns (avg_loss, epoch_duration_seconds).
# ---------------------------------------------------------------------------
def train_one_epoch(context_tower, paper_tower, loader, optimizer,
                    scaler, all_paper_feats, device, temperature, epoch):
    context_tower.train()
    paper_tower.train()
    t0, total_loss, n_batches = time.time(), 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        batch_feats = {k: v[batch["cited_paper_id"]].to(device)
                       for k, v in all_paper_feats.items()}

        with torch.amp.autocast('cuda'):
            loss = infonce_loss(
                context_tower(batch["input_ids"].to(device),
                              batch["attention_mask"].to(device)),
                paper_tower(batch_feats),
                temperature,
            )

        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"\n[WARN] NaN/Inf at epoch {epoch} batch {n_batches} — skipping.")
            optimizer.zero_grad()
            n_batches += 1
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(context_tower.parameters()) + list(paper_tower.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_val
        n_batches  += 1
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / max(n_batches, 1), time.time() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    print("=" * 60)
    print(f"  Experiment : {cfg['experiment_name']}")
    for k, v in cfg.items():
        print(f"    {k:20s} = {v}")
    print("=" * 60 + "\n")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])

    device    = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")
    data_root = os.path.expanduser(cfg["data_root"])
    ckpt_dir  = os.path.join(os.path.expanduser(cfg["output_dir"]), cfg["experiment_name"])
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # --- Datasets ---
    datasets = build_datasets(
        all_contexts_path = os.path.join(data_root, "all_contexts.json"),
        node_index_path   = os.path.join(data_root, "node_index.json"),
        max_length        = cfg["max_length"],
        seed              = cfg["seed"],
        split_path        = os.path.join(data_root, "split_uris.json"),
    )

    # Standard shuffled DataLoader — no GroupAwareBatchSampler needed since
    # this dataset has no multi-citation grouping (see module docstring).
    train_loader = DataLoader(datasets["train"],
                              batch_size   = cfg["batch_size"],
                              shuffle      = True,
                              collate_fn   = lcr_collate_fn,
                              num_workers  = 4,
                              pin_memory   = True)

    val_loader   = DataLoader(datasets["val"], batch_size=cfg["batch_size"],
                              shuffle=False, collate_fn=lcr_collate_fn,
                              num_workers=4, pin_memory=True)

    test_loader  = DataLoader(datasets["test"], batch_size=cfg["batch_size"],
                              shuffle=False, collate_fn=lcr_collate_fn,
                              num_workers=4, pin_memory=True)

    # --- Metapath features (precomputed, stored on disk, never change) ---
    print("Loading metapath features ...")
    all_paper_feats = {}
    for key in cfg["feat_keys"]:
        all_paper_feats[key] = torch.load(
            os.path.join(data_root, f"feat_{key}.pt"), map_location="cpu")
        print(f"  feat_{key}: {all_paper_feats[key].shape}")

    # --- Full candidate pool (ALL papers — no corpus/external split) ---
    with open(os.path.join(data_root, "node_index.json"), encoding="utf-8") as f:
        node_index = json.load(f)
    n_papers = len(node_index["paper"])
    all_candidate_ids = torch.arange(n_papers, dtype=torch.long)
    print(f"Total candidates: {len(all_candidate_ids):,}\n")

    # --- Models ---
    paper_tower = PaperTower(
        feat_keys=cfg["feat_keys"],
        nfeat=768,
        num_heads = cfg["num_heads"],
        hidden=cfg["hidden"],
        embed_dim=cfg["embed_dim"],
        n_fp_layers=cfg["n_fp_layers"],
        dropout=cfg["dropout"],
        input_drop=cfg["input_drop"],
        att_drop=cfg["att_drop"],
        act=cfg["act"],
        residual=cfg["residual"],
        use_mlp = cfg["use_mlp"],
    ).to(device)

    context_tower = ContextTower(
        embed_dim=cfg["embed_dim"],
        dropout=cfg["input_drop"],
        scibert_model_name=cfg["scibert_model_name"],
    ).to(device)

    print(f"PaperTower params  : {sum(p.numel() for p in paper_tower.parameters()):,}")
    print(f"ContextTower params: {sum(p.numel() for p in context_tower.parameters()):,}\n")

    optimizer = torch.optim.Adam([
        *context_tower.get_param_groups(cfg["lr_scibert"], cfg["lr_head"]),
        {"params": paper_tower.parameters(), "lr": cfg["lr_paper"]},
    ])
    scaler = torch.amp.GradScaler('cuda')

    # --- Training loop ---
    best_mrr, best_epoch, patience_ctr = 0.0, 0, 0
    history        = []
    test_history   = []
    test_curve_path = os.path.join(ckpt_dir, "test_curve.json")
    training_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):

        candidate_embs = build_candidate_index(
            paper_tower, all_paper_feats, all_candidate_ids, device)

        train_loss, epoch_secs = train_one_epoch(
            context_tower, paper_tower, train_loader, optimizer,
            scaler, all_paper_feats, device, cfg["temperature"], epoch)

        print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | time={epoch_secs:.1f}s", end="")

        metrics = evaluate(context_tower, val_loader, candidate_embs, all_candidate_ids, device)
        print(f" | R@1={metrics.get('Recall@1',0):.4f}"
              f" R@10={metrics.get('Recall@10',0):.4f}"
              f" MRR={metrics.get('MRR',0):.4f}"
              f" nDCG@10={metrics.get('nDCG@10',0):.4f}"
              f" (n={metrics.get('n_queries',0):,})")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "epoch_secs": round(epoch_secs, 2), **metrics})

        val_mrr = metrics.get("MRR", 0.0)
        if val_mrr > best_mrr:
            best_mrr, best_epoch, patience_ctr = val_mrr, epoch, 0
            torch.save({"epoch": epoch, "paper_tower": paper_tower.state_dict(),
                        "context_tower": context_tower.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_mrr": best_mrr, "metrics": metrics, "config": cfg},
                       os.path.join(ckpt_dir, "best_model.pt"))
            print(f"  ✓ New best MRR={best_mrr:.4f} — saved")
        else:
            patience_ctr += 1
            print(f"  (patience {patience_ctr}/{cfg['patience']})")
            if patience_ctr >= cfg["patience"]:
                print(f"\nEarly stopping — best MRR={best_mrr:.4f} at epoch {best_epoch}")
                break

        if epoch % cfg["test_eval_every"] == 0 or epoch == cfg["epochs"]:
            post_train_candidate_embs = build_candidate_index(
                paper_tower, all_paper_feats, all_candidate_ids, device)

            test_metrics = evaluate(context_tower, test_loader, post_train_candidate_embs,
                                    all_candidate_ids, device)
            test_history.append({"epoch": epoch, **test_metrics})

            with open(test_curve_path, "w") as f:
                json.dump({"experiment_name": cfg["experiment_name"],
                           "test_eval_every": cfg["test_eval_every"],
                           "history": test_history}, f, indent=2)

    total = time.time() - training_start

    summary = {
        "best_mrr":                best_mrr,
        "best_epoch":              best_epoch,
        "total_training_time_h":   round(total / 3600, 4),
        "epochs_run":              len(history),
        "history":                 history,
        "test_history":            test_history,
    }
    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest MRR={best_mrr:.4f} at epoch {best_epoch}")
    print(f"Total training time: {total/3600:.2f}h ({total/60:.1f} min)")


if __name__ == "__main__":
    main()