"""
train.py
--------
Training script for the LCR two-tower retrieval model.
Reads all hyperparameters from a JSON config file passed via --config.

Architecture:
    ContextTower  — SciBERT (fine-tuned) → [B, embed_dim]
    PaperTower    — SeHGNN metapath fusion → [N, embed_dim]

Loss: InfoNCE over in-batch negatives
    - Anchor   : context embedding
    - Positive : cited paper embedding
    - Negatives: all other papers in the batch

Evaluation metrics: Recall@K (K=1,5,10,20), MRR, nDCG@10
    - Corpus index is a FROZEN SNAPSHOT built before each epoch's training step.

Usage:
    python train.py --config configs/exp01_PP_768.json
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.expanduser("~/HGNN")
sys.path.insert(0, os.path.join(ROOT, "paper_tower"))
sys.path.insert(0, os.path.join(ROOT, "context_tower"))
sys.path.insert(0, os.path.join(ROOT, "shared", "data_prep"))

from paper_tower.model   import PaperTower
from context_tower.model import ContextTower
from dataset import build_datasets, lcr_collate_fn



# ---------------------------------------------------------------------------
# Config — load JSON, no defaults, every key must be present
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# InfoNCE contrastive loss
# For each context in the batch, the cited paper is the positive and the
# other (batch_size - 1) papers are the negatives.
# ---------------------------------------------------------------------------
def infonce_loss(ctx_emb, paper_emb, temperature):
    logits   = torch.matmul(ctx_emb, paper_emb.T) / temperature
    labels   = torch.arange(logits.size(0), device=logits.device)
    loss_c2p = F.cross_entropy(logits,   labels)   # context → paper
    loss_p2c = F.cross_entropy(logits.T, labels)   # paper   → context
    return (loss_c2p + loss_p2c) / 2.0


# ---------------------------------------------------------------------------
# Build frozen candidate index
# Runs PaperTower over all 26K papers ONCE before training starts each epoch.
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
    return torch.cat(embs, dim=0)   # [N_candidates, embed_dim]


# ---------------------------------------------------------------------------
# Evaluation — ranks all 26K candidates for each val context
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(context_tower, loader, candidate_embs, candidate_ids, device):
    context_tower.eval()

    cand_dev      = candidate_embs.to(device)
    global_to_pos = {gid: pos for pos, gid in enumerate(candidate_ids.tolist())}

    recall_hits = {k: 0 for k in [1, 5, 10, 20]}
    mrr_sum, ndcg_sum, n_queries, n_skipped = 0.0, 0.0, 0, 0

    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        ctx_emb   = context_tower(batch["input_ids"].to(device),
                                  batch["attention_mask"].to(device))
        sims      = torch.matmul(ctx_emb, cand_dev.T)

        for i, cited_id in enumerate(batch["cited_paper_id"].tolist()):
            if cited_id not in global_to_pos:
                n_skipped += 1
                continue
            pos  = global_to_pos[cited_id]
            rank = int((sims[i] > sims[i][pos]).sum().item()) + 1
            for k in [1, 5, 10, 20]:
                if rank <= k:
                    recall_hits[k] += 1
            mrr_sum  += 1.0 / rank
            ndcg_sum += 1.0 / math.log2(rank + 1)
            n_queries += 1

    if n_queries == 0:
        print("  [WARN] No valid queries.")
        return {}

    metrics = {f"Recall@{k}": recall_hits[k] / n_queries for k in [1, 5, 10, 20]}
    metrics.update({"MRR": mrr_sum / n_queries, "nDCG@10": ndcg_sum / n_queries,
                    "n_queries": n_queries, "n_skipped": n_skipped})
    return metrics


# ---------------------------------------------------------------------------
# One training epoch
# For each batch: encode 64 contexts + their 64 cited papers → InfoNCE loss
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
        # Skip corrupted batches — NaN/Inf gradients would permanently damage weights
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

    # Print config so every run is self-documented in the logs
    print("=" * 60)
    print(f"  Experiment : {cfg['experiment_name']}")
    for k, v in cfg.items():
        print(f"    {k:20s} = {v}")
    print("=" * 60 + "\n")

    # Seeding
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])

    device    = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")
    data_root = os.path.expanduser(cfg["data_root"])
    ckpt_dir  = os.path.join(os.path.expanduser(cfg["output_dir"]), cfg["experiment_name"])
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config copy into checkpoint folder for full reproducibility
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # --- Datasets ---
    datasets = build_datasets(
        all_contexts_path = os.path.join(data_root, "all_contexts.json"),
        node_index_path   = os.path.join(data_root, "node_index.json"),
        max_length        = cfg["max_length"],
        seed              = cfg["seed"],
    )

    # Batch formulation using the PyTorch default randomSampler
    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"], #
                              shuffle=True,  collate_fn=lcr_collate_fn,
                              num_workers=4, pin_memory=True)
    
    
    val_loader   = DataLoader(datasets["val"], batch_size=64,
                              shuffle=False, collate_fn=lcr_collate_fn,
                              num_workers=4, pin_memory=True)

    # --- Metapath features (precomputed, stored on disk, never change) ---
    print("Loading metapath features ...")
    all_paper_feats = {}
    for key in cfg["feat_keys"]:
        all_paper_feats[key] = torch.load(
            os.path.join(data_root, f"feat_{key}.pt"), map_location="cpu")
        print(f"  feat_{key}: {all_paper_feats[key].shape}")

    # --- Full candidate pool (corpus + external papers) ---
    all_candidate_ids = torch.cat([
        torch.load(os.path.join(data_root, "corpus_ids.pt"),   map_location="cpu"),
        torch.load(os.path.join(data_root, "external_ids.pt"), map_location="cpu"),
    ]).unique()
    print(f"Total candidates: {len(all_candidate_ids):,}\n")

    # --- Models ---
    paper_tower = PaperTower(
        feat_keys=cfg["feat_keys"], nfeat=768,
        hidden=cfg["hidden"], embed_dim=cfg["embed_dim"],
        n_fp_layers=cfg["n_fp_layers"],
        dropout=cfg["dropout"], input_drop=cfg["input_drop"],
    ).to(device)

    context_tower = ContextTower(
        embed_dim=cfg["embed_dim"], dropout=cfg["input_drop"],
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
    training_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):

        # Build frozen index before training — used only for evaluation
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

        # Checkpoint if improved
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

    # Save training history
    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    total = time.time() - training_start
    print(f"\nBest MRR={best_mrr:.4f} at epoch {best_epoch}")
    print(f"Total training time: {total/60:.1f} min ({total:.0f}s)")


if __name__ == "__main__":
    main()