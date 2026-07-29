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
    python train.py --config configs/P+PP/no_MLP/experience.json
    
    nohup python train.py --config configs/P+PP/MLP_500/experience.json >myoutfile 2>&1 &
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
sys.path.insert(0, os.path.join(ROOT, "shared", "data_prep"))

from paper_tower.model   import PaperTower
from context_tower.model import ContextTower
from dataset import build_datasets, lcr_collate_fn
from group_aware_sampler import GroupAwareBatchSampler



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
# This is safe against the "sibling rows with identical context_text" problem
# (a multi-citation marker like [1,2,3] producing several records that share
# context_text but point to different cited papers) only because
# GroupAwareBatchSampler already guarantees no two records from the same
# context_group_id ever co-occur in a batch. The loss itself doesn't need to
# know group_ids exist — the sampler upstream has made every row's negatives
# trustworthy by construction.
# ---------------------------------------------------------------------------
def infonce_loss(ctx_emb, paper_emb, temperature):
    # Compute similarity between every context and every paper in the batch → [B, B]
    logits = torch.matmul(ctx_emb, paper_emb.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_c2p = F.cross_entropy(logits,   labels)  # context → paper: Makes each context pull its correct paper closer than other papers. Good for "given context, rank papers"
    loss_p2c = F.cross_entropy(logits.T, labels)  # paper → context: Makes each paper pull its correct context closer than other contexts. This forces paper embeddings to spread out and stay distinct.

    return (loss_c2p + loss_p2c) / 2.0


# ---------------------------------------------------------------------------
# Build frozen candidate index
# Runs PaperTower over all 26K papers ONCE before training starts each epoch.
# The resulting matrix is fixed for the entire evaluation of that epoch —
# weights keep updating during training but the index does not.
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_candidate_index(paper_tower, all_paper_feats, candidate_ids, device):
    # Save training state so we can restore it after — eval() disables dropout
    # which is required for deterministic embeddings during evaluation
    was_training = paper_tower.training
    paper_tower.eval()

    embs = []
    ids  = candidate_ids.tolist()

    # Encode all 26K candidate contexts in chunks of 512 
    # The 512 contexts then moved to CPU  memory to free up GPU memory for the next chunk to avoid GPU memory overflow
    # append the CPU embeddings to a python list and concatenate all chunks to form the final candidate embedding matrix
    for start in tqdm(range(0, len(ids), 512), desc="  Building index", leave=False):
        batch_ids   = ids[start : start + 512]
        # Look up precomputed metapath features for this chunk from the fixed disk tensors
        batch_feats = {k: v[batch_ids].to(device) for k, v in all_paper_feats.items()}
        # Move embeddings back to CPU immediately to free GPU memory
        embs.append(paper_tower(batch_feats).cpu())

    # Restore training state before returning
    paper_tower.train(was_training)

    # Concatenate all chunks → [N_candidates, embed_dim]
    return torch.cat(embs, dim=0)


# ---------------------------------------------------------------------------
# Evaluation — ranks all 26K candidates for each val context
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(context_tower, loader, candidate_embs, candidate_ids, device):
    """
    Citations sharing the same context_group_id (citation_type == "multiple")
    are treated as ONE evaluation unit (= one s in S_{P_c}).

    Example: a context group with 3 true citations ranked [1, 4, 9]
    contributes Recall@1 = 1/1 = 1.0 (only 1 slot possible, capped),
    Recall@5 = 2/3 (2 hits out of 3 possible, since 3 <= 5).
    This group counts as ONE query, same weight as a group with only
    1 true citation -- that's what "macro-averaged" means here.
    """
    context_tower.eval()

    # Move the full candidate pool embeddings to GPU/CPU device once,
    # so we don't repeatedly transfer them inside the loop.
    cand_dev = candidate_embs.to(device)

    # --- Zero-vector filtering -------------------------------------------
    # Papers whose PaperTower embedding is exactly all-zero (e.g. external
    # papers under a PP-only config, which have no outgoing adj_PP edges)
    # all get identical similarity scores against every query. Since rank
    # is "1 + count of candidates strictly ahead", these tied zero-vector
    # papers can NEVER outrank anything -- they're invisible to the rank
    # calculation but still silently occupy pool slots. Left in, they let
    # a zero-vector TRUE paper win rank=1 "for free". So we drop them from
    # the candidate pool before ranking anything.
    nonzero_mask = cand_dev.abs().sum(dim=1) != 0
    n_zero_candidates = (~nonzero_mask).sum().item()
    cand_dev = cand_dev[nonzero_mask]

    # Map each candidate paper's global ID -> its row index in cand_dev.
    # Needed because "cited_paper_id" in the batch is a global ID, but
    # sims[i] is indexed positionally (0..num_candidates-1).

    # FILTERED cand_dev. Global IDs whose embedding was zero are simply
    # absent from this dict now, so any cited_id lookup against it will
    # fail gracefully.
    kept_ids = [gid for gid, keep in zip(candidate_ids.tolist(), nonzero_mask.tolist()) if keep]
    global_to_pos = {gid: pos for pos, gid in enumerate(kept_ids)}

    # Standard cutoffs to report Recall at.
    k_values = [1, 5, 10, 20]

    # automatically create a list for each new group_id, so we can append ranks to it without pre-initializing
    # group_ranks[group_id] = list of ranks (one per true citation) for
    # that context. E.g. group_ranks[59328] = [1, 4, 9] means this context
    # has 3 true cited papers, found at ranks 1, 4, and 9 in the candidate
    # ranking. Rows with citation_type == "single" just end up as groups
    # of size 1.
    group_ranks = defaultdict(list)

    # Counts cited papers that don't exist in the candidate pool at all
    # (e.g. external papers not indexed) -- these are skipped, not
    # counted as misses, so they don't unfairly punish the model.
    n_skipped = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cited_ids = batch["cited_paper_id"].tolist()
            group_ids = batch["context_group_id"].tolist()

            # Encode this batch of context strings into embeddings.
            # NOTE: for "multiple" citation type, several rows in the
            # batch will share the SAME context text (and same
            # input_ids), because the same sentence cites several
            # papers. Their ctx_emb (and therefore sims) will be
            # IDENTICAL -- only the target paper (cited_id) differs
            # per row. This is why identical-context rows can produce
            # tied similarity scores / tied ranks (see chat discussion
            # on why rank ties like [2, 2] can occur).
            ctx_emb = context_tower(input_ids, attention_mask)

            # Similarity of each context embedding against every
            # candidate paper embedding -> shape (batch_size, num_candidates).
            sims = torch.matmul(ctx_emb, cand_dev.T)

            for i, cited_id in enumerate(cited_ids):
                if cited_id not in global_to_pos:
                    # True citation isn't in our candidate pool
                    # (e.g. filtered out / external paper) -- can't
                    # evaluate rank for it, so skip.
                    # sanity check: this shouldn't happen
                    n_skipped += 1
                    continue

                # Position of the true cited paper within the
                # candidate similarity vector for this row.
                pos = global_to_pos[cited_id]

                # --- Step 1: Compare true paper's score against every candidate's score ---
                # sims[i] = similarity scores of context i against ALL candidate papers
                # sims[i][pos] = similarity score of the TRUE cited paper
                true_score = sims[i][pos]

                # Boolean vector: True wherever a candidate scored STRICTLY higher
                # than the true paper (ties do NOT count as "higher")
                beats_true_paper = sims[i] > true_score

                # --- Step 2: Count how many candidates scored strictly higher ---
                # This tells us how many papers would be ranked ABOVE the true paper
                num_candidates_ahead = beats_true_paper.sum()   # still a tensor
                num_candidates_ahead = num_candidates_ahead.item()  # convert to plain Python number
                num_candidates_ahead = int(num_candidates_ahead)    # make sure it's an int

                # --- Step 3: Convert count into a 1-indexed rank ---
                # If 0 candidates beat it -> it's rank 1 (best possible)
                # If 3 candidates beat it -> it's rank 4
                rank = num_candidates_ahead + 1

                # Store this rank under its context group, so all true
                # citations belonging to the same context/sentence get
                # aggregated together as ONE evaluation unit.
                group_ranks[group_ids[i]].append(rank)

    # Running sums, one per k, to be macro-averaged over groups at the end.
    recall_hits = {k: 0.0 for k in k_values}
    mrr_sum, ndcg_sum, n_queries = 0.0, 0.0, 0

    for group_id, ranks in group_ranks.items():
        # ranks holds one entry per TRUE citation belonging to this group
        # (appended earlier via group_ranks[group_id].append(rank)).
        #
        # - single-citation context  -> ranks = [rank1]              -> len = 1
        # - multi-citation context   -> ranks = [rank1, rank2, rank3] -> len = 3
        #   (one rank per co-cited paper in that sentence, e.g. "[12, 13, 14]")
        #
        # So n_papers is NOT the number of candidates compared against --
        # it's the number of ground-truth correct papers for this one query/group.
        # needed below to correctly cap the Recall@K denominator via
        # min(n_papers, k) -- e.g. a group with 3 true citations shouldn't be
        # penalized for only having 1 possible hit slot when k=1.
        n_papers = len(ranks)

        for k in k_values:
            # Cap the denominator at k: you can never get more than k
            # hits out of k slots, even if the group has more true
            # citations than k. This matches min(|cit(G(s))|, n) 
            # WITHOUT this cap, groups with
            # many true citations would have their recall artificially
            # deflated (e.g. 1 hit / 5 true citations = 0.2 instead of
            # the correct 1 hit / 1 possible slot = 1.0 at k=1).
            denom = min(n_papers, k)

            # Count how many of this group's true citations landed
            # within the top-k.
            hits = 0
            for r in ranks:
                if r <= k:
                    hits = hits + 1

            # Add this group's Recall@k contribution. Divided by
            # n_queries later to get the macro-average across groups.
            recall_hits[k] += hits / denom

        # MRR: average of 1/rank across this group's true citations,
        # then this per-group average gets summed here and divided by
        # n_queries below (i.e. groups are NOT weighted by their size).
        mrr_sum += sum(1.0 / r for r in ranks) / n_papers

        # nDCG@10 style discount: average of 1/log2(rank+1) across this
        # group's true citations, same within-group-then-across-groups
        # averaging as MRR.
        ndcg_sum += sum(1.0 / math.log2(r + 1) for r in ranks) / n_papers

        # Each group (not each raw citation row) counts as ONE query.
        n_queries += 1

    # Final macro-averages: divide every accumulated sum by the number
    # of GROUPS (context units), not the number of raw citation rows.
    metrics = {f"Recall@{k}": recall_hits[k] / n_queries for k in k_values}
    metrics.update({"MRR": mrr_sum / n_queries, "nDCG@10": ndcg_sum / n_queries,
                "n_queries": n_queries, "n_skipped": n_skipped,
                "n_zero_candidates_dropped": n_zero_candidates})
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
        split_path        = os.path.join(data_root, "split_uris.json"),
    )

    # Batch formulation using GroupAwareBatchSampler:
    # guarantees no two records sharing the same context_group_id (i.e. same
    # citation context from a multi-citation marker like [1,2,3]) ever land in
    # the SAME batch. Combined with the plain single-positive infonce_loss
    # above, this keeps every row's negatives trustworthy — a sibling row's
    # true paper never has to be scored as a negative against an identical
    # context embedding.
    train_batch_sampler = GroupAwareBatchSampler(
        group_ids  = [r.context_group_id for r in datasets["train"].records],
        batch_size = cfg["batch_size"],
        seed       = cfg["seed"],
    )
    train_loader = DataLoader(datasets["train"],
                              batch_sampler = train_batch_sampler,  # replaces batch_size + shuffle
                              collate_fn    = lcr_collate_fn,
                              num_workers   = 4,
                              pin_memory    = True)
    
    
    val_loader   = DataLoader(datasets["val"], batch_size=cfg["batch_size"],
                              shuffle=False, collate_fn=lcr_collate_fn,
                              num_workers=4, pin_memory=True)

    # Test loader — used only for the periodic curve-tracing eval every
    # cfg["test_eval_every"] epochs (kept separate from val, which drives
    # early stopping/checkpointing every epoch).
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

    # --- Full candidate pool (corpus + external papers) ---
    all_candidate_ids = torch.cat([
        torch.load(os.path.join(data_root, "corpus_ids.pt"),   map_location="cpu"),
        torch.load(os.path.join(data_root, "external_ids.pt"), map_location="cpu"),
    ]).unique()
    print(f"Total candidates: {len(all_candidate_ids):,}\n")

    # --- Models ---
    paper_tower = PaperTower(
        feat_keys=cfg["feat_keys"],
        nfeat=768, # the dimension output by SciBERT-based features when precomputeing metapath features 
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

        # Reseed the sampler each epoch so batch composition varies (mirrors shuffle=True)
        train_batch_sampler.set_epoch(epoch)

        # Build frozen index before training — used only for evaluationR
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

        # --- Periodic test-set evaluation (curve tracing) ---
        # Uses a FRESH candidate index built from the model's weights AFTER this
        # epoch's training step — unlike val above, which intentionally evaluates
        # against the frozen pre-epoch snapshot. Costs one extra full pass over
        # the candidate pool every test_eval_every epochs.
        if epoch % cfg["test_eval_every"] == 0 or epoch == cfg["epochs"]:
            post_train_candidate_embs = build_candidate_index(
                paper_tower, all_paper_feats, all_candidate_ids, device)

            test_metrics = evaluate(context_tower, test_loader, post_train_candidate_embs,
                                    all_candidate_ids, device)
            test_history.append({"epoch": epoch, **test_metrics})

            # Write after every eval, not just at the end, so the curve survives
            # early stopping / crashes / preemption.
            with open(test_curve_path, "w") as f:
                json.dump({"experiment_name": cfg["experiment_name"],
                           "test_eval_every": cfg["test_eval_every"],
                           "history": test_history}, f, indent=2)

    total = time.time() - training_start

    
    summary = {
        "best_mrr":                best_mrr,
        "best_epoch":              best_epoch,
        "total_training_time_h":   round(total / 3600, 4),  # e.g. 1.2345 hours
        "epochs_run":              len(history),
        "history":                 history,
        "test_history":            test_history,  # same data as test_curve.json, duplicated here for convenience
    }
    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest MRR={best_mrr:.4f} at epoch {best_epoch}")
    print(f"Total training time: {total/3600:.2f}h ({total/60:.1f} min)")


if __name__ == "__main__":
    main()