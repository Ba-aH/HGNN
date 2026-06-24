"""
train.py
--------
Training script for the LCR two-tower retrieval model.

Architecture:
    ContextTower  — SciBERT (fine-tuned) + projection head → [B, embed_dim]
    PaperTower    — SeHGNN metapath fusion + projection head → [N, embed_dim]

Loss: InfoNCE over in-batch negatives
    - Anchor  : context embedding
    - Positive: cited paper embedding
    - Negatives: all other papers in the batch

Evaluation metrics: Recall@K (K=1,5,10,20), MRR, nDCG@10
    - At eval time, the corpus index is a FROZEN SNAPSHOT of PaperTower embeddings
      computed BEFORE the epoch's training step begins (option 3 fix).
    - This prevents the live model from being evaluated against the same feature
      tensors it was trained on, which caused memorisation collapse at epoch ~16.

Usage:
    python train.py \
    --data_root ~/HGNN/shared/data_prep \
    --output_dir ~/HGNN/checkpoints \
    --epochs 20 \
    --batch_size 64 \
    --embed_dim 256
"""

import os
import sys
import json
import math
import copy
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — import from sibling directories
# ---------------------------------------------------------------------------
ROOT = os.path.expanduser("~/HGNN")
sys.path.insert(0, os.path.join(ROOT, "paper_tower"))
sys.path.insert(0, os.path.join(ROOT, "context_tower"))
sys.path.insert(0, os.path.join(ROOT, "shared", "data_prep"))

from paper_tower.model   import PaperTower      # noqa: E402
from context_tower.model import ContextTower    # noqa: E402
from dataset import build_datasets, lcr_collate_fn  # noqa: E402


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

def infonce_loss(context_emb: torch.Tensor, paper_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE loss over in-batch negatives.

    Parameters
    ----------
    context_emb : [B, embed_dim]  L2-normalised context embeddings
    paper_emb   : [B, embed_dim]  L2-normalised paper embeddings for the
                                  positive (cited) paper of each context
    temperature : float           Learnable or fixed temperature (0.07 is
                                  the SimCLR/CLIP default)

    Returns
    -------
    scalar loss
    """
    # Similarity matrix [B, B] — dot product = cosine sim (both L2-normalised)
    logits = torch.matmul(context_emb, paper_emb.T) / temperature  # [B, B]

    # Diagonal entries are the positives
    labels = torch.arange(logits.size(0), device=logits.device)

    # Cross-entropy from both directions (symmetric)
    loss_c2p = F.cross_entropy(logits,   labels)   # context → paper
    loss_p2c = F.cross_entropy(logits.T, labels)   # paper → context

    return (loss_c2p + loss_p2c) / 2.0


# ---------------------------------------------------------------------------
# FIX (option 3): Frozen corpus index builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_corpus_index(
    paper_tower:       nn.Module,
    all_paper_feats:   dict,          # {key: Tensor[N_total, 768]} on CPU
    corpus_ids:        torch.Tensor,  # [N_corpus] global paper IDs
    device:            torch.device,
    batch_size_papers: int = 512,
) -> torch.Tensor:
    """
    Encodes the full corpus with a SNAPSHOT of paper_tower taken at the
    moment this function is called, then returns the resulting embedding
    matrix [N_corpus, embed_dim] on CPU.

    This is the core of option 3:
      - Called ONCE at the start of each epoch, before any gradient updates.
      - evaluate() receives this pre-built tensor and never calls paper_tower
        again, so the live model weights cannot influence the corpus index.
      - Between training batches the weights shift, but the index stays frozen
        for the entire evaluation of that epoch — giving an honest ranking signal.

    The function:
      1. Sets paper_tower to eval mode (no dropout, no batchnorm running stats).
      2. Encodes corpus papers in chunks to stay within GPU memory.
      3. Returns the matrix to CPU so evaluate() can move rows to GPU as needed.
      4. Restores paper_tower.training to whatever it was on entry (train mode
         is set back by train_one_epoch at the top of the next epoch).
    """
    was_training = paper_tower.training
    paper_tower.eval()

    corpus_ids_list = corpus_ids.tolist()
    corpus_embs = []

    for start in range(0, len(corpus_ids_list), batch_size_papers):
        batch_ids = corpus_ids_list[start : start + batch_size_papers]
        batch_feats = {
            k: v[batch_ids].to(device)
            for k, v in all_paper_feats.items()
        }
        emb = paper_tower(batch_feats)          # [b, embed_dim]
        corpus_embs.append(emb.cpu())

    paper_tower.train(was_training)             # restore original mode

    return torch.cat(corpus_embs, dim=0)        # [N_corpus, embed_dim] on CPU


# ---------------------------------------------------------------------------
# Evaluation — now receives a pre-built frozen corpus index
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    context_tower:  nn.Module,
    loader:         DataLoader,
    corpus_embs:    torch.Tensor,     # FIX: [N_corpus, embed_dim] — frozen snapshot, on CPU
    corpus_ids:     torch.Tensor,     # [N_corpus] global paper IDs
    device:         torch.device,
    k_values:       list = [1, 5, 10, 20],
) -> dict:
    """
    Full ranking evaluation over the validation or test set.

    CHANGED from original:
      - paper_tower is no longer a parameter. The corpus index was already built
        by build_corpus_index() before training started for this epoch.
      - corpus_embs is a frozen [N_corpus, embed_dim] tensor passed in directly.
      - This function only runs ContextTower, which is correct: at inference time
        you encode the query with ContextTower and rank against a static index.

    For each context query:
        1. Encode context → context_emb  [B, embed_dim]
        2. Rank all corpus papers by cosine similarity against frozen corpus_embs
        3. Find rank of the ground-truth cited paper
        4. Compute Recall@K, MRR, nDCG@10
    """
    context_tower.eval()

    # Move frozen corpus index to device once for the whole eval pass
    corpus_embs_dev = corpus_embs.to(device)   # [N_corpus, embed_dim]

    # Build a mapping: global_paper_id → position in corpus_embs
    corpus_ids_list = corpus_ids.tolist()
    global_to_corpus_pos = {gid: pos for pos, gid in enumerate(corpus_ids_list)}

    # --- Evaluate each context query ---
    recall_hits = {k: 0 for k in k_values}
    mrr_sum   = 0.0
    ndcg_sum  = 0.0
    n_queries = 0
    n_skipped = 0   # cited paper not in corpus (external paper)

    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cited_ids      = batch["cited_paper_id"].tolist()

        ctx_emb = context_tower(input_ids, attention_mask)  # [B, embed_dim]

        # Cosine similarities against frozen corpus
        sims = torch.matmul(ctx_emb, corpus_embs_dev.T)     # [B, N_corpus]

        for i, cited_id in enumerate(cited_ids):
            if cited_id not in global_to_corpus_pos:
                n_skipped += 1
                continue

            pos = global_to_corpus_pos[cited_id]
            sim_row = sims[i]   # [N_corpus]

            # Rank of the positive (1-indexed, lower is better)
            rank = int((sim_row > sim_row[pos]).sum().item()) + 1

            for k in k_values:
                if rank <= k:
                    recall_hits[k] += 1

            mrr_sum  += 1.0 / rank
            ndcg_sum += 1.0 / math.log2(rank + 1)
            n_queries += 1

    if n_queries == 0:
        print("  [WARN] No valid queries found in eval set.")
        return {}

    metrics = {f"Recall@{k}": recall_hits[k] / n_queries for k in k_values}
    metrics["MRR"]        = mrr_sum  / n_queries
    metrics["nDCG@10"]    = ndcg_sum / n_queries
    metrics["n_queries"]  = n_queries
    metrics["n_skipped"]  = n_skipped

    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    context_tower: nn.Module,
    paper_tower:   nn.Module,
    loader:        DataLoader,
    optimizer:     torch.optim.Optimizer,
    scaler:        torch.amp.GradScaler,
    all_paper_feats: dict,
    device:        torch.device,
    temperature:   float,
    epoch:         int,
) -> float:
    context_tower.train()
    paper_tower.train()

    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cited_ids      = batch["cited_paper_id"]          # [B] cpu LongTensor

        # Fetch metapath features for the cited papers in this batch
        batch_paper_feats = {
            k: v[cited_ids].to(device)
            for k, v in all_paper_feats.items()
        }

        # FIX (bug 2): compute loss BEFORE backward so we can skip NaN batches
        # without poisoning the weights with a corrupted gradient.
        with torch.amp.autocast('cuda'):
            ctx_emb   = context_tower(input_ids, attention_mask)   # [B, embed_dim]
            paper_emb = paper_tower(batch_paper_feats)             # [B, embed_dim]
            loss      = infonce_loss(ctx_emb, paper_emb, temperature)

        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            # Skip this batch entirely — zero_grad so no stale grads accumulate
            print(f"\n[WARN] NaN/Inf loss at epoch {epoch} batch {n_batches}, skipping.")
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

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LCR Two-Tower Training")

    # Paths
    parser.add_argument("--data_root",     default="~/HGNN/shared/data_prep")
    parser.add_argument("--output_dir",    default="~/HGNN/checkpoints")

    # Model
    parser.add_argument("--embed_dim",     type=int,   default=256)
    parser.add_argument("--hidden",        type=int,   default=512)
    parser.add_argument("--n_fp_layers",   type=int,   default=2)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--input_drop",    type=float, default=0.1)
    parser.add_argument("--temperature",   type=float, default=0.07)

    # Training
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--max_length",    type=int,   default=256)
    parser.add_argument("--lr_scibert",    type=float, default=2e-6)
    parser.add_argument("--lr_head",       type=float, default=1e-4)
    parser.add_argument("--lr_paper",      type=float, default=1e-3)
    parser.add_argument("--patience",      type=int,   default=7)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--gpu",           type=int,   default=0)

    # Evaluation Ran full in every N epochs
    parser.add_argument("--eval_every",    type=int,   default=1)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root  = os.path.expanduser(args.data_root)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(output_dir, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints → {ckpt_dir}")

    # --- Dataset ---
    datasets = build_datasets(
        all_contexts_path = os.path.join(data_root, "all_contexts.json"),
        node_index_path   = os.path.join(data_root, "node_index.json"),
        max_length        = args.max_length,
        seed              = args.seed,
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size  = args.batch_size,
        shuffle     = True,
        collate_fn  = lcr_collate_fn,
        num_workers = 4,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size  = 64,
        shuffle     = False,
        collate_fn  = lcr_collate_fn,
        num_workers = 4,
        pin_memory  = True,
    )

    # --- Load precomputed metapath feature tensors ---
    print("Loading metapath feature tensors ...")
    feat_keys = ["P", "PP", "PCP"]
    all_paper_feats = {}
    for key in feat_keys:
        path = os.path.join(data_root, f"feat_{key}.pt")
        all_paper_feats[key] = torch.load(path, map_location="cpu")
        print(f"  feat_{key}: {all_paper_feats[key].shape}")

    # --- Load corpus_ids (papers to rank against at eval time) ---
    corpus_ids = torch.load(os.path.join(data_root, "corpus_ids.pt"), map_location="cpu")
    print(f"Corpus size (for ranking): {len(corpus_ids):,}")

    # --- Build models ---
    paper_tower = PaperTower(
        feat_keys   = feat_keys,
        nfeat       = 768,
        hidden      = args.hidden,
        embed_dim   = args.embed_dim,
        n_fp_layers = args.n_fp_layers,
        dropout     = args.dropout,
        input_drop  = args.input_drop,
    ).to(device)

    context_tower = ContextTower(
        embed_dim = args.embed_dim,
        dropout   = args.input_drop,
    ).to(device)

    print(f"\nPaperTower  params: {sum(p.numel() for p in paper_tower.parameters()):,}")
    print(f"ContextTower params: {sum(p.numel() for p in context_tower.parameters()):,}")

    # --- Optimiser: three param groups with different LRs ---
    optimizer = torch.optim.Adam([
        *context_tower.get_param_groups(args.lr_scibert, args.lr_head),
        {"params": paper_tower.parameters(), "lr": args.lr_paper},
    ])

    scaler = torch.amp.GradScaler('cuda')

    # --- Training loop ---
    best_mrr     = 0.0
    best_epoch   = 0
    patience_ctr = 0
    history      = []

    print(f"\nStarting training for {args.epochs} epochs ...\n")

    for epoch in range(1, args.epochs + 1):

        # Build the frozen corpus index BEFORE training this epoch.
        #
        # Why before, not after?
        #   - We want the eval to reflect the model state at the START of the epoch,
        #     i.e. what the model learned up to but not including the current epoch's
        #     gradient updates.
        #   - This is the honest signal: "given what I know so far, how well can I rank?"
        #
        # Memory note: ~1,770 papers × 256 dims × 4 bytes ≈ 1.8 MB. Negligible.
        if epoch % args.eval_every == 0:
            print(f"  Building frozen corpus index (epoch {epoch} snapshot) ...")
            corpus_embs = build_corpus_index(
                paper_tower     = paper_tower,
                all_paper_feats = all_paper_feats,
                corpus_ids      = corpus_ids,
                device          = device,
            )
            print(f"  Corpus index shape: {corpus_embs.shape}  (on CPU)")

        # --- Train ---
        train_loss = train_one_epoch(
            context_tower   = context_tower,
            paper_tower     = paper_tower,
            loader          = train_loader,
            optimizer       = optimizer,
            scaler          = scaler,
            all_paper_feats = all_paper_feats,
            device          = device,
            temperature     = args.temperature,
            epoch           = epoch,
        )

        log = {"epoch": epoch, "train_loss": train_loss}
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f}", end="")

        # --- Evaluation against the frozen snapshot ---
        if epoch % args.eval_every == 0:
            metrics = evaluate(
                context_tower = context_tower,
                loader        = val_loader,
                corpus_embs   = corpus_embs,   # frozen snapshot, not live paper_tower
                corpus_ids    = corpus_ids,
                device        = device,
            )
            log.update(metrics)

            print(
                f" | R@1={metrics.get('Recall@1', 0):.4f}"
                f" R@5={metrics.get('Recall@5', 0):.4f}"
                f" R@10={metrics.get('Recall@10', 0):.4f}"
                f" MRR={metrics.get('MRR', 0):.4f}"
                f" nDCG@10={metrics.get('nDCG@10', 0):.4f}"
                f" (n={metrics.get('n_queries', 0):,})"
            )

            # --- Checkpoint on best val MRR ---
            val_mrr = metrics.get("MRR", 0.0)
            if val_mrr > best_mrr:
                best_mrr   = val_mrr
                best_epoch = epoch
                patience_ctr = 0

                ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
                torch.save({
                    "epoch":         epoch,
                    "paper_tower":   paper_tower.state_dict(),
                    "context_tower": context_tower.state_dict(),
                    "optimizer":     optimizer.state_dict(),
                    "val_mrr":       best_mrr,
                    "metrics":       metrics,
                    "args":          vars(args),
                }, ckpt_path)
                print(f"  ✓ New best MRR={best_mrr:.4f} — checkpoint saved")
            else:
                patience_ctr += 1
                print(f"  (no improvement, patience {patience_ctr}/{args.patience})")

            if patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} — best MRR={best_mrr:.4f} at epoch {best_epoch}")
                break
        else:
            print()

        history.append(log)

    # --- Save training history ---
    history_path = os.path.join(ckpt_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print(f"Best val MRR={best_mrr:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()