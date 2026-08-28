"""
dataset.py
----------
PyTorch Dataset + train/val/test split for the LCR two-tower model.

ADAPTED for the current dataset: no citation_type / context_group_id fields
exist in all_contexts.json (confirmed: adj_PCP has 0 non-zero entries, i.e.
every citation node cites exactly one paper — there is no multi-citation
grouping to track). CitationRecord and the filter logic below have been
simplified accordingly. GroupAwareBatchSampler is NOT needed for this
dataset — use a standard shuffled DataLoader instead (see note in
train.py section below).

Each sample:
    context_text  : str                 — citing passage (input to ContextTower)
    cited_paper_id: int                 — global integer ID of the cited paper
                                          (index into feat_P / feat_PP / feat_PCCon tensors)

Source file: all_contexts.json
    [
      {
        "context":     <str>,
        "cited_uri":   "https://citekg.org/resource/paper/<hash>",
        "citing_uri":  "https://citekg.org/resource/paper/<hash>",
        "citing_idx":  <int>
      },
      ...
    ]

Filters applied:
    - context must be non-empty after stripping
    - citing_uri must be non-empty
    - cited_uri must exist in node_index["paper"]

Split: loaded from a frozen split_uris.json (see data_split.py). Falls back
to a deterministic seed-based shuffle only if split_path is not provided or
doesn't exist yet.
"""

import os
import json
import random
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class CitationRecord:
    context_text:      str
    cited_paper_id:    int   # global integer index into feat tensors
    cited_uri:         str   # kept for debugging / evaluation
    citing_uri:        str


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LCRDataset(Dataset):  # list of pairs (context_text, cited_paper_id)
    """
    Parameters
    ----------
    records : list[CitationRecord]
    tokenizer : transformers tokenizer
    max_length : int
        Maximum token length for SciBERT (hard cap 512).
    """

    def __init__(
        self,
        records: List[CitationRecord],
        tokenizer,
        max_length: int = 256,
    ):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        enc = self.tokenizer(
            rec.context_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,          # collate_fn handles padding
            return_tensors=None,    # return plain lists; collate pads to batch max
        )

        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "cited_paper_id": rec.cited_paper_id,
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def lcr_collate_fn(batch):
    """
    Pads input_ids and attention_mask to the longest sequence in the batch.
    Returns:
        input_ids      : LongTensor [B, max_seq_len]
        attention_mask : LongTensor [B, max_seq_len]
        cited_paper_id : LongTensor [B]
    """
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids      = []
    attention_mask = []
    cited_ids      = []

    for x in batch:
        seq_len = len(x["input_ids"])
        pad_len = max_len - seq_len

        input_ids.append(x["input_ids"] + [0] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        cited_ids.append(x["cited_paper_id"])

    return {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "cited_paper_id": torch.tensor(cited_ids,       dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_datasets(
    all_contexts_path: str,
    node_index_path:   str,
    tokenizer_name:    str  = "allenai/scibert_scivocab_uncased",
    max_length:        int  = 256,
    train_ratio:       float = 0.8,
    val_ratio:         float = 0.1,
    seed:              int   = 42,
    split_path:        Optional[str] = None,
) -> dict:
    """
    Loads all_contexts.json, filters, maps URIs → integer IDs,
    splits into train/val/test, and returns LCRDataset objects.

    Split behavior
    --------------
    If `split_path` is given and the file exists, the train/val/test
    paper_uri lists are loaded directly from it (see data_split.py) — the
    seed/shuffle logic below is skipped entirely, so the split cannot
    silently drift even if all_contexts.json / node_index.json change later.

    If `split_path` is given but the file does NOT exist yet, this function
    falls back to the seed-based shuffle-and-split below, and then WRITES
    the resulting citing_uri lists to `split_path`.

    If `split_path` is None, the split is recomputed every call from
    (seed, current data files).

    Returns
    -------
    {
        "train": LCRDataset,
        "val":   LCRDataset,
        "test":  LCRDataset,
        "tokenizer": tokenizer,
        "n_papers":  int,
    }
    """
    # --- Load node index → {uri: int_id} ---
    print(f"Loading node index from {node_index_path} ...")
    with open(node_index_path, encoding="utf-8") as f:
        node_index = json.load(f)
    paper_uri_to_id = node_index["paper"]
    n_papers = len(paper_uri_to_id)
    print(f"  {n_papers:,} paper nodes in KG")

    # --- Load and filter records ---
    print(f"Loading contexts from {all_contexts_path} ...")
    with open(all_contexts_path, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  {len(raw):,} raw records")

    records = []
    skipped = 0
    for item in raw:
        cited_uri    = item.get("cited_uri", "")
        context_text = item.get("context", "").strip()
        citing_uri   = item.get("citing_uri", "")

        if not context_text:
            skipped += 1
            continue
        if not citing_uri:
            skipped += 1
            continue
        if cited_uri not in paper_uri_to_id:
            skipped += 1
            continue

        records.append(CitationRecord(
            context_text   = context_text,
            cited_paper_id = paper_uri_to_id[cited_uri],
            cited_uri      = cited_uri,
            citing_uri     = citing_uri,
        ))

    print(f"  {len(records):,} records kept, {skipped:,} skipped")

    # --- Split: frozen file (preferred) or deterministic shuffle (fallback) ---
    used_frozen_split = False
    if split_path is not None and os.path.exists(split_path):
        print(f"Loading frozen split from {split_path} ...")
        with open(split_path, encoding="utf-8") as f:
            frozen = json.load(f)
        train_uris = set(frozen["train"])
        val_uris   = set(frozen["val"])
        test_uris  = set(frozen["test"])
        used_frozen_split = True
    else:
        citing_uris = list({r.citing_uri for r in records})
        rng = random.Random(seed)
        rng.shuffle(citing_uris)

        n_uris       = len(citing_uris)
        n_train_uris = int(n_uris * train_ratio)
        n_val_uris   = int(n_uris * val_ratio)

        train_uris = set(citing_uris[:n_train_uris])
        val_uris   = set(citing_uris[n_train_uris : n_train_uris + n_val_uris])
        test_uris  = set(citing_uris[n_train_uris + n_val_uris :])

        if split_path is not None:
            print(f"No frozen split found at {split_path} — computing it now "
                  f"and saving it so future runs are locked to it.")
            os.makedirs(os.path.dirname(os.path.abspath(split_path)) or ".", exist_ok=True)
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump({
                    "train": sorted(train_uris),
                    "val":   sorted(val_uris),
                    "test":  sorted(test_uris),
                    "meta": {
                        "seed": seed,
                        "train_ratio": train_ratio,
                        "val_ratio": val_ratio,
                        "n_kept": len(records),
                        "n_skipped": skipped,
                    },
                }, f, indent=2)

    train_records = [r for r in records if r.citing_uri in train_uris]
    val_records   = [r for r in records if r.citing_uri in val_uris]
    test_records  = [r for r in records if r.citing_uri in test_uris]

    split_source = "frozen file" if used_frozen_split else "seed-based shuffle"
    print(f"  Split ({split_source}) → train {len(train_records):,} / "
          f"val {len(val_records):,} / test {len(test_records):,}")

    # --- Tokenizer ---
    print(f"Loading tokenizer ({tokenizer_name}) ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return {
        "train":     LCRDataset(train_records, tokenizer, max_length),
        "val":       LCRDataset(val_records,   tokenizer, max_length),
        "test":      LCRDataset(test_records,  tokenizer, max_length),
        "tokenizer": tokenizer,
        "n_papers":  n_papers,
    }
