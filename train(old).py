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
    - At eval time, PaperTower encodes the FULL corpus once (precomputed)
    - Each context query is ranked against all corpus papers

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
    # Takes a batch of B context embedding and B cited paper embeddings -> Compute [B,B] similarity matrix
    # -> run cross-entropy in both directions (context -> paper) & (paper -> context) => averages both    
    # InfoNCE : measure the discrepancy (difference) between the predicted probability distribution of words and the actual distribution observed in the training data
    # the original SeHGNN use Cross-entropy loss (because their objective is classification)
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
# Evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    context_tower: nn.Module,
    paper_tower:   nn.Module,
    loader:        DataLoader,
    all_paper_feats: dict,        # {key: Tensor[N_total, 768]}  on CPU
    corpus_ids:    torch.Tensor,  # [N_corpus] global paper IDs to rank against
    device:        torch.device,
    k_values:      list = [1, 5, 10, 20],
    batch_size_papers: int = 512,
) -> dict:
    """
    Full ranking evaluation over the validation or test set.

    For each context query:
        1. Encode context → context_emb [1, embed_dim]
        2. Rank all corpus papers by cosine similarity
        3. Find rank of the ground-truth cited paper
        4. Compute Recall@K, MRR, nDCG@10

    Returns dict of metric_name → float.
    """
    context_tower.eval()
    paper_tower.eval()

    # --- Precompute all corpus paper embeddings ---
    # Only encode corpus papers (those with abstracts / real features)
    # against which we rank at inference time
    print("  Precomputing corpus paper embeddings ...")
    corpus_embs = []
    corpus_ids_list = corpus_ids.tolist()

    for start in range(0, len(corpus_ids_list), batch_size_papers):
        batch_ids = corpus_ids_list[start : start + batch_size_papers]
        batch_feats = {
            k: v[batch_ids].to(device)
            for k, v in all_paper_feats.items()
        }
        emb = paper_tower(batch_feats)   # [b, embed_dim]
        corpus_embs.append(emb.cpu())

    corpus_embs = torch.cat(corpus_embs, dim=0)   # [N_corpus, embed_dim]
    corpus_embs = corpus_embs.to(device)

    # Build a mapping: global_paper_id → position in corpus_embs 
    # the ground-truth cited_paper_id in the batch is a global ID
    global_to_corpus_pos = {gid: pos for pos, gid in enumerate(corpus_ids_list)}

    # --- Evaluate each context query ---
    recall_hits = {k: 0 for k in k_values}
    mrr_sum   = 0.0
    ndcg_sum  = 0.0
    n_queries = 0
    n_skipped = 0   # cited paper not in corpus (external paper)

    # system splits valid/test queries into small batches 
    # for every query in a batch ( multiple citation context)
    # verify each passage belongs to wich paper
    # for each batch it queries each cotnext and check if the context belong the correct cited paper meaning this sentence is referencing this paper
    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cited_ids      = batch["cited_paper_id"].tolist()

        ctx_emb = context_tower(input_ids, attention_mask)  # [B, embed_dim]

        # Cosine similarities against all corpus papers
        sims = torch.matmul(ctx_emb, corpus_embs.T)   # [B, N_corpus]

        for i, cited_id in enumerate(cited_ids):
            if cited_id not in global_to_corpus_pos:
                # Cited paper is external — skip for ranking eval
                n_skipped += 1
                continue
                
            # This finds the position (index) of that correct passage inside the full corpus.
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
    metrics["MRR"]     = mrr_sum  / n_queries
    metrics["nDCG@10"] = ndcg_sum / n_queries
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
    for batch in pbar: # fetch paper features
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cited_ids      = batch["cited_paper_id"]          # [B] cpu LongTensor

        # Fetch metapath features for the cited papers in this batch
        batch_paper_feats = {
            k: v[cited_ids].to(device)
            for k, v in all_paper_feats.items()
        }

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            # Forward pass
            # Both towers produce L2-normalised embeddings of shape [B, 256]. InfoNCE then builds a [B, B] cosine similarity matrix
            ctx_emb   = context_tower(input_ids, attention_mask)   # [B, embed_dim]
            paper_emb = paper_tower(batch_paper_feats)             # [B, embed_dim]
            loss      = infonce_loss(ctx_emb, paper_emb, temperature)

        # every thing after this is made to avoid underflow in float16 gradients added (fixed NaN issue) (backward pass)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(context_tower.parameters()) + list(paper_tower.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"\n[WARN] NaN/Inf loss at batch {n_batches}, skipping.")
            n_batches += 1
            continue

        total_loss += loss_val
        n_batches  += 1
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / n_batches


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
    parser.add_argument("--dropout",       type=float, default=0.5)
    parser.add_argument("--input_drop",    type=float, default=0.1)
    parser.add_argument("--temperature",   type=float, default=0.07)

    # Training
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--max_length",    type=int,   default=256)
    parser.add_argument("--lr_scibert",    type=float, default=2e-6)
    parser.add_argument("--lr_head",       type=float, default=1e-3)
    parser.add_argument("--lr_paper",      type=float, default=1e-3)
    parser.add_argument("--patience",      type=int,   default=5,
                        help="Early stopping patience (epochs without val MRR improvement)")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--gpu",           type=int,   default=0)

    # Evaluation
    parser.add_argument("--eval_every",    type=int,   default=1,
                        help="Run full evaluation every N epochs")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Reproducibility ---
    # Guarantees that the data split, weight initialization, and dropout patterns are identical across runs with the same seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root  = os.path.expanduser(args.data_root)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(output_dir, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints → {ckpt_dir}")

    # --- Dataset ---
    # calls build_datasets from shared/data_prep/dataset.py
    # 
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

    n_papers = datasets["n_papers"]

    # --- Load precomputed metapath feature tensors ---
    # the precomputed embedding of each paper
    print("Loading metapath feature tensors ...")
    feat_keys = ["P", "PP", "PCP"]
    all_paper_feats = {}
    for key in feat_keys:
        path = os.path.join(data_root, f"feat_{key}.pt")
        all_paper_feats[key] = torch.load(path, map_location="cpu")
        print(f"  feat_{key}: {all_paper_feats[key].shape}")

    # --- Load corpus_ids (papers to rank against at eval time) ---
    # corpus_ids is a list of paper IDs that are valid candidates for ranking.
    # exclude papers without any feature embeddings (no abstract and no citation context)
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
    best_mrr      = 0.0
    best_epoch    = 0
    patience_ctr  = 0
    history       = []

    print(f"\nStarting training for {args.epochs} epochs ...\n")

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(
            context_tower    = context_tower,
            paper_tower      = paper_tower,
            loader           = train_loader,
            optimizer        = optimizer,
            scaler           = scaler,
            all_paper_feats  = all_paper_feats,
            device           = device,
            temperature      = args.temperature,
            epoch            = epoch,
        )

        log = {"epoch": epoch, "train_loss": train_loss}
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f}", end="")

        # --- Evaluation ---
        if epoch % args.eval_every == 0:
            metrics = evaluate(
                context_tower    = context_tower,
                paper_tower      = paper_tower,
                loader           = val_loader,
                all_paper_feats  = all_paper_feats,
                corpus_ids       = corpus_ids,
                device           = device,
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
                    "epoch":          epoch,
                    "paper_tower":    paper_tower.state_dict(),
                    "context_tower":  context_tower.state_dict(),
                    "optimizer":      optimizer.state_dict(),
                    "val_mrr":        best_mrr,
                    "metrics":        metrics,
                    "args":           vars(args),
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