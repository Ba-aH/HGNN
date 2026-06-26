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
# InfoNCE loss 
# ---------------------------------------------------------------------------
def infonce_loss(context_emb: torch.Tensor, paper_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = torch.matmul(context_emb, paper_emb.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_c2p = F.cross_entropy(logits, labels) # loss (context  - paper)
    loss_p2c = F.cross_entropy(logits.T, labels) # loss (paper - context)
    return (loss_c2p + loss_p2c) / 2.0


# ---------------------------------------------------------------------------
# Build frozen candidate index (corpus + external) papers
# At the start of each epoch, before any training happens, 
# it runs every paper in the corpus through the current paper_tower and stores the resulting embeddings in a big matrix. 
# That matrix is then used as the fixed "search index" during evaluation.
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_candidate_index(
    paper_tower:       nn.Module,
    all_paper_feats:   dict,
    candidate_ids:     torch.Tensor,  
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

    candidate_ids_list = candidate_ids.tolist()
    candidate_embs = []

    for start in tqdm(range(0, len(candidate_ids_list), batch_size_papers),
                      desc="  Building candidate index", leave=False):
        batch_ids = candidate_ids_list[start : start + batch_size_papers]
        batch_feats = {
            k: v[batch_ids].to(device)
            for k, v in all_paper_feats.items()
        }
        emb = paper_tower(batch_feats)
        candidate_embs.append(emb.cpu())

    paper_tower.train(was_training)
    return torch.cat(candidate_embs, dim=0)  # [N_candidates, embed_dim] on CPU


# ---------------------------------------------------------------------------
# Evaluation 
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    context_tower:   nn.Module,
    loader:          DataLoader,
    candidate_embs:  torch.Tensor,    # [N_candidates, embed_dim] — frozen
    candidate_ids:   torch.Tensor,    # All paper IDs (corpus + external)
    device:          torch.device,
    k_values:        list = [1, 5, 10, 20],
) -> dict:
    context_tower.eval()

    candidate_embs_dev = candidate_embs.to(device)
    candidate_ids_list = candidate_ids.tolist()
    global_to_pos = {gid: pos for pos, gid in enumerate(candidate_ids_list)}

    recall_hits = {k: 0 for k in k_values}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    n_queries = 0
    n_skipped = 0

    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cited_ids = batch["cited_paper_id"].tolist()

        ctx_emb = context_tower(input_ids, attention_mask)  # [B, embed_dim]

        sims = torch.matmul(ctx_emb, candidate_embs_dev.T)  # [B, N_candidates]

        for i, cited_id in enumerate(cited_ids):
            if cited_id not in global_to_pos:
                n_skipped += 1
                continue

            pos = global_to_pos[cited_id]
            sim_row = sims[i]

            rank = int((sim_row > sim_row[pos]).sum().item()) + 1

            for k in k_values:
                if rank <= k:
                    recall_hits[k] += 1

            mrr_sum += 1.0 / rank
            ndcg_sum += 1.0 / math.log2(rank + 1)
            n_queries += 1

    if n_queries == 0:
        print("  [WARN] No valid queries found.")
        return {}

    metrics = {f"Recall@{k}": recall_hits[k] / n_queries for k in k_values}
    metrics["MRR"]       = mrr_sum / n_queries
    metrics["nDCG@10"]   = ndcg_sum / n_queries
    metrics["n_queries"] = n_queries
    metrics["n_skipped"] = n_skipped

    return metrics


# ---------------------------------------------------------------------------
# Training loop (unchanged logic)
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
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False) # Wraps the DataLoader in a progress bar so you see a live update in the terminal as batches are processed.
    for batch in pbar: # for each batch a dict of (input_ids, attention_maask, cited_paper_id) for 64 citation records
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cited_ids = batch["cited_paper_id"]

        batch_paper_feats = { # grap paper features for each batch 64 papers
            k: v[cited_ids].to(device)
            for k, v in all_paper_feats.items()
        }

        with torch.amp.autocast('cuda'):  # Forward pass through both towers
            ctx_emb = context_tower(input_ids, attention_mask) # this encodes the 64 contexts(citing passages)
            paper_emb = paper_tower(batch_paper_feats) # this encode the cited papers
            loss = infonce_loss(ctx_emb, paper_emb, temperature) # each context should be ranked #1 agains the other 64 papers


        # We check for NaN/Inf loss to catch numerical instability (common in contrastive training + AMP).
        # If we don’t skip it, the NaN/Inf gradients will corrupt the optimizer and usually ruin the entire training run.
        # The code safely skips the bad batch, clears gradients, and continues.
        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"\n[WARN] NaN/Inf loss at epoch {epoch} batch {n_batches}, skipping.")
            optimizer.zero_grad()
            n_batches += 1
            continue


        optimizer.zero_grad() # clear leftover gradient from previous batch
        scaler.scale(loss).backward() # run backpropagation 
                                    # calculates how much each weight in both towers contributed to the loss and stores that in each parameter's .grad.

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(context_tower.parameters()) + list(paper_tower.parameters()),
            max_norm=1.0,
        )

        # Update the weights:
            # context tower: - all SciBERT weights being updated
            #                - the projection linear layer + LayerNorm that maps 768 -> 256 
            # paper tower: - all SciBERT weights being updated
            #              - the projection linear layer + LayerNorm that maps 768 -> 256
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_val
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Argument parser 
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="LCR Two-Tower Training (Full Candidates)")
    parser.add_argument("--data_root",     default="~/HGNN/shared/data_prep")
    parser.add_argument("--output_dir",    default="~/HGNN/checkpoints")

    parser.add_argument("--embed_dim",     type=int,   default=256)
    parser.add_argument("--hidden",        type=int,   default=512) # Hidden layer size inside the PaperTower
    parser.add_argument("--n_fp_layers",   type=int,   default=2)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--input_drop",    type=float, default=0.1)
    parser.add_argument("--temperature",   type=float, default=0.07)

    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--max_length",    type=int,   default=256) # Maximum token length for the citation context
    parser.add_argument("--lr_scibert",    type=float, default=2e-6) # applied inside context module and it's extremely low LR because scibert already trained on scientific papers and im not retraining it we simply using it to generate embeddings 
    parser.add_argument("--lr_head",       type=float, default=1e-4) #Applied to the projection head (the part that turns SciBERT output into the final embedding)
    parser.add_argument("--lr_paper",      type=float, default=1e-3) # applied inside the PaperModule and it's high becuase we are training the HGNN from scratch and we are not using a pretrained model
    parser.add_argument("--patience",      type=int,   default=7)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--gpu",           type=int,   default=0)
    parser.add_argument("--eval_every",    type=int,   default=1) # evaluate on every epoch

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = os.path.expanduser(args.data_root)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(output_dir, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints → {ckpt_dir}")

    # --- Datasets ---
    datasets = build_datasets(
        all_contexts_path = os.path.join(data_root, "all_contexts.json"),
        node_index_path   = os.path.join(data_root, "node_index.json"),
        max_length        = args.max_length,
        seed              = args.seed,
    )

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                              collate_fn=lcr_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(datasets["val"], batch_size=64, shuffle=False,
                            collate_fn=lcr_collate_fn, num_workers=4, pin_memory=True)

    # --- Features ---
    print("Loading metapath feature tensors ...")
    feat_keys = ["P", "PP"]
    all_paper_feats = {}
    for key in feat_keys:
        path = os.path.join(data_root, f"feat_{key}.pt")
        all_paper_feats[key] = torch.load(path, map_location="cpu")
        print(f"  feat_{key}: {all_paper_feats[key].shape}")

    # --- Load corpus + external → full candidate pool ---
    corpus_ids = torch.load(os.path.join(data_root, "corpus_ids.pt"), map_location="cpu")
    external_ids = torch.load(os.path.join(data_root, "external_ids.pt"), map_location="cpu")

    all_candidate_ids = torch.cat([corpus_ids, external_ids]).unique()
    print(f"Corpus size     : {len(corpus_ids):,}")
    print(f"External size   : {len(external_ids):,}")
    print(f"Total candidates: {len(all_candidate_ids):,} (will be used for ranking)\n")

    # --- Models ---
    paper_tower = PaperTower(
        feat_keys=feat_keys, nfeat=768, hidden=args.hidden,
        embed_dim=args.embed_dim, n_fp_layers=args.n_fp_layers,
        dropout=args.dropout, input_drop=args.input_drop,
    ).to(device)

    context_tower = ContextTower(embed_dim=args.embed_dim, dropout=args.input_drop).to(device)

    print(f"PaperTower params : {sum(p.numel() for p in paper_tower.parameters()):,}")
    print(f"ContextTower params: {sum(p.numel() for p in context_tower.parameters()):,}")

    optimizer = torch.optim.Adam([
        *context_tower.get_param_groups(args.lr_scibert, args.lr_head),
        {"params": paper_tower.parameters(), "lr": args.lr_paper},
    ])

    scaler = torch.amp.GradScaler('cuda')

    # --- Training loop ---
    best_mrr = 0.0
    best_epoch = 0
    patience_ctr = 0
    history = []

    print(f"Starting training for {args.epochs} epochs ...\n")

    for epoch in range(1, args.epochs + 1):
        # Build frozen index for ALL candidates
        if epoch % args.eval_every == 0:
            print(f"  Building frozen candidate index (epoch {epoch}) ...")
            candidate_embs = build_candidate_index(
                paper_tower=paper_tower,
                all_paper_feats=all_paper_feats,
                candidate_ids=all_candidate_ids,
                device=device,
            )
            print(f"  Candidate index shape: {candidate_embs.shape}")

        # --- Train ---
        train_loss = train_one_epoch(
            context_tower=context_tower,
            paper_tower=paper_tower,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            all_paper_feats=all_paper_feats,
            device=device,
            temperature=args.temperature,
            epoch=epoch,
        )

        log = {"epoch": epoch, "train_loss": train_loss}
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f}", end="")

        # --- Evaluation ---
        if epoch % args.eval_every == 0:
            metrics = evaluate(
                context_tower=context_tower,
                loader=val_loader,
                candidate_embs=candidate_embs,
                candidate_ids=all_candidate_ids,
                device=device,
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

            val_mrr = metrics.get("MRR", 0.0)
            if val_mrr > best_mrr:
                best_mrr = val_mrr
                best_epoch = epoch
                patience_ctr = 0

                ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "paper_tower": paper_tower.state_dict(),
                    "context_tower": context_tower.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_mrr": best_mrr,
                    "metrics": metrics,
                    "args": vars(args),
                }, ckpt_path)
                print(f"  ✓ New best MRR={best_mrr:.4f} — saved")
            else:
                patience_ctr += 1
                print(f"  (patience {patience_ctr}/{args.patience})")

            if patience_ctr >= args.patience:
                print(f"\nEarly stopping — best MRR={best_mrr:.4f} at epoch {best_epoch}")
                break
        else:
            print()

        history.append(log)

    # Save history
    history_path = os.path.join(ckpt_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print(f"Best val MRR = {best_mrr:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()"""
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
# InfoNCE loss 
# ---------------------------------------------------------------------------
def infonce_loss(context_emb: torch.Tensor, paper_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = torch.matmul(context_emb, paper_emb.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_c2p = F.cross_entropy(logits, labels) # loss (context  - paper)
    loss_p2c = F.cross_entropy(logits.T, labels) # loss (paper - context)
    return (loss_c2p + loss_p2c) / 2.0


# ---------------------------------------------------------------------------
# Build frozen candidate index (corpus + external) papers
# At the start of each epoch, before any training happens, 
# it runs every paper in the corpus through the current paper_tower and stores the resulting embeddings in a big matrix. 
# That matrix is then used as the fixed "search index" during evaluation.
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_candidate_index(
    paper_tower:       nn.Module,
    all_paper_feats:   dict,
    candidate_ids:     torch.Tensor,  
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

    candidate_ids_list = candidate_ids.tolist()
    candidate_embs = []

    for start in tqdm(range(0, len(candidate_ids_list), batch_size_papers),
                      desc="  Building candidate index", leave=False):
        batch_ids = candidate_ids_list[start : start + batch_size_papers]
        batch_feats = {
            k: v[batch_ids].to(device)
            for k, v in all_paper_feats.items()
        }
        emb = paper_tower(batch_feats)
        candidate_embs.append(emb.cpu())

    paper_tower.train(was_training)
    return torch.cat(candidate_embs, dim=0)  # [N_candidates, embed_dim] on CPU


# ---------------------------------------------------------------------------
# Evaluation 
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    context_tower:   nn.Module,
    loader:          DataLoader,
    candidate_embs:  torch.Tensor,    # [N_candidates, embed_dim] — frozen
    candidate_ids:   torch.Tensor,    # All paper IDs (corpus + external)
    device:          torch.device,
    k_values:        list = [1, 5, 10, 20],
) -> dict:
    context_tower.eval()

    candidate_embs_dev = candidate_embs.to(device)
    candidate_ids_list = candidate_ids.tolist()
    global_to_pos = {gid: pos for pos, gid in enumerate(candidate_ids_list)}

    recall_hits = {k: 0 for k in k_values}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    n_queries = 0
    n_skipped = 0

    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cited_ids = batch["cited_paper_id"].tolist()

        ctx_emb = context_tower(input_ids, attention_mask)  # [B, embed_dim]

        sims = torch.matmul(ctx_emb, candidate_embs_dev.T)  # [B, N_candidates]

        for i, cited_id in enumerate(cited_ids):
            if cited_id not in global_to_pos:
                n_skipped += 1
                continue

            pos = global_to_pos[cited_id]
            sim_row = sims[i]

            rank = int((sim_row > sim_row[pos]).sum().item()) + 1

            for k in k_values:
                if rank <= k:
                    recall_hits[k] += 1

            mrr_sum += 1.0 / rank
            ndcg_sum += 1.0 / math.log2(rank + 1)
            n_queries += 1

    if n_queries == 0:
        print("  [WARN] No valid queries found.")
        return {}

    metrics = {f"Recall@{k}": recall_hits[k] / n_queries for k in k_values}
    metrics["MRR"]       = mrr_sum / n_queries
    metrics["nDCG@10"]   = ndcg_sum / n_queries
    metrics["n_queries"] = n_queries
    metrics["n_skipped"] = n_skipped

    return metrics


# ---------------------------------------------------------------------------
# Training loop (unchanged logic)
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
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False) # Wraps the DataLoader in a progress bar so you see a live update in the terminal as batches are processed.
    for batch in pbar: # for each batch a dict of (input_ids, attention_maask, cited_paper_id) for 64 citation records
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cited_ids = batch["cited_paper_id"]

        batch_paper_feats = { # grap paper features for each batch 64 papers
            k: v[cited_ids].to(device)
            for k, v in all_paper_feats.items()
        }

        with torch.amp.autocast('cuda'):  # Forward pass through both towers
            ctx_emb = context_tower(input_ids, attention_mask) # this encodes the 64 contexts(citing passages)
            paper_emb = paper_tower(batch_paper_feats) # this encode the cited papers
            loss = infonce_loss(ctx_emb, paper_emb, temperature) # each context should be ranked #1 agains the other 64 papers


        # We check for NaN/Inf loss to catch numerical instability (common in contrastive training + AMP).
        # If we don’t skip it, the NaN/Inf gradients will corrupt the optimizer and usually ruin the entire training run.
        # The code safely skips the bad batch, clears gradients, and continues.
        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"\n[WARN] NaN/Inf loss at epoch {epoch} batch {n_batches}, skipping.")
            optimizer.zero_grad()
            n_batches += 1
            continue


        optimizer.zero_grad() # clear leftover gradient from previous batch
        scaler.scale(loss).backward() # run backpropagation 
                                    # calculates how much each weight in both towers contributed to the loss and stores that in each parameter's .grad.

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(context_tower.parameters()) + list(paper_tower.parameters()),
            max_norm=1.0,
        )

        # Update the weights:
            # context tower: - all SciBERT weights being updated
            #                - the projection linear layer + LayerNorm that maps 768 -> 256 
            # paper tower: - all SciBERT weights being updated
            #              - the projection linear layer + LayerNorm that maps 768 -> 256
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_val
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Argument parser 
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="LCR Two-Tower Training (Full Candidates)")
    parser.add_argument("--data_root",     default="~/HGNN/shared/data_prep")
    parser.add_argument("--output_dir",    default="~/HGNN/checkpoints")

    parser.add_argument("--embed_dim",     type=int,   default=768)
    parser.add_argument("--hidden",        type=int,   default=768) # Hidden layer size inside the PaperTower
    parser.add_argument("--n_fp_layers",   type=int,   default=2)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--input_drop",    type=float, default=0.1)
    parser.add_argument("--temperature",   type=float, default=0.07)

    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--max_length",    type=int,   default=256) # Maximum token length for the citation context
    parser.add_argument("--lr_scibert",    type=float, default=2e-6) # applied inside context module and it's extremely low LR because scibert already trained on scientific papers and im not retraining it we simply using it to generate embeddings 
    parser.add_argument("--lr_head",       type=float, default=1e-4) #Applied to the projection head (the part that turns SciBERT output into the final embedding)
    parser.add_argument("--lr_paper",      type=float, default=1e-3) # applied inside the PaperModule and it's high becuase we are training the HGNN from scratch and we are not using a pretrained model
    parser.add_argument("--patience",      type=int,   default=7)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--gpu",           type=int,   default=0)
    parser.add_argument("--eval_every",    type=int,   default=1) # evaluate on every epoch

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = os.path.expanduser(args.data_root)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(output_dir, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints → {ckpt_dir}")

    # --- Datasets ---
    datasets = build_datasets(
        all_contexts_path = os.path.join(data_root, "all_contexts.json"),
        node_index_path   = os.path.join(data_root, "node_index.json"),
        max_length        = args.max_length,
        seed              = args.seed,
    )

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                              collate_fn=lcr_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(datasets["val"], batch_size=64, shuffle=False,
                            collate_fn=lcr_collate_fn, num_workers=4, pin_memory=True)

    # --- Features ---
    print("Loading metapath feature tensors ...")
    feat_keys = ["P", "PP"]
    all_paper_feats = {}
    for key in feat_keys:
        path = os.path.join(data_root, f"feat_{key}.pt")
        all_paper_feats[key] = torch.load(path, map_location="cpu")
        print(f"  feat_{key}: {all_paper_feats[key].shape}")

    # --- Load corpus + external → full candidate pool ---
    corpus_ids = torch.load(os.path.join(data_root, "corpus_ids.pt"), map_location="cpu")
    external_ids = torch.load(os.path.join(data_root, "external_ids.pt"), map_location="cpu")

    all_candidate_ids = torch.cat([corpus_ids, external_ids]).unique()
    print(f"Corpus size     : {len(corpus_ids):,}")
    print(f"External size   : {len(external_ids):,}")
    print(f"Total candidates: {len(all_candidate_ids):,} (will be used for ranking)\n")

    # --- Models ---
    paper_tower = PaperTower(
        feat_keys=feat_keys, nfeat=768, hidden=args.hidden,
        embed_dim=args.embed_dim, n_fp_layers=args.n_fp_layers,
        dropout=args.dropout, input_drop=args.input_drop,
    ).to(device)

    context_tower = ContextTower(embed_dim=args.embed_dim, dropout=args.input_drop).to(device)

    print(f"PaperTower params : {sum(p.numel() for p in paper_tower.parameters()):,}")
    print(f"ContextTower params: {sum(p.numel() for p in context_tower.parameters()):,}")

    optimizer = torch.optim.Adam([
        *context_tower.get_param_groups(args.lr_scibert, args.lr_head),
        {"params": paper_tower.parameters(), "lr": args.lr_paper},
    ])

    scaler = torch.amp.GradScaler('cuda')

    # --- Training loop ---
    best_mrr = 0.0
    best_epoch = 0
    patience_ctr = 0
    history = []

    print(f"Starting training for {args.epochs} epochs ...\n")

    for epoch in range(1, args.epochs + 1):
        # Build frozen index for ALL candidates
        if epoch % args.eval_every == 0:
            print(f"  Building frozen candidate index (epoch {epoch}) ...")
            candidate_embs = build_candidate_index(
                paper_tower=paper_tower,
                all_paper_feats=all_paper_feats,
                candidate_ids=all_candidate_ids,
                device=device,
            )
            print(f"  Candidate index shape: {candidate_embs.shape}")

        # --- Train ---
        train_loss = train_one_epoch(
            context_tower=context_tower,
            paper_tower=paper_tower,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            all_paper_feats=all_paper_feats,
            device=device,
            temperature=args.temperature,
            epoch=epoch,
        )

        log = {"epoch": epoch, "train_loss": train_loss}
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f}", end="")

        # --- Evaluation ---
        if epoch % args.eval_every == 0:
            metrics = evaluate(
                context_tower=context_tower,
                loader=val_loader,
                candidate_embs=candidate_embs,
                candidate_ids=all_candidate_ids,
                device=device,
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

            val_mrr = metrics.get("MRR", 0.0)
            if val_mrr > best_mrr:
                best_mrr = val_mrr
                best_epoch = epoch
                patience_ctr = 0

                ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "paper_tower": paper_tower.state_dict(),
                    "context_tower": context_tower.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_mrr": best_mrr,
                    "metrics": metrics,
                    "args": vars(args),
                }, ckpt_path)
                print(f"  ✓ New best MRR={best_mrr:.4f} — saved")
            else:
                patience_ctr += 1
                print(f"  (patience {patience_ctr}/{args.patience})")

            if patience_ctr >= args.patience:
                print(f"\nEarly stopping — best MRR={best_mrr:.4f} at epoch {best_epoch}")
                break
        else:
            print()

        history.append(log)

    # Save history
    history_path = os.path.join(ckpt_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print(f"Best val MRR = {best_mrr:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()