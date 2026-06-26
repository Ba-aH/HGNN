"""
shared/data_prep/dataset.py
---------------------------
PyTorch Dataset + train/val/test split for the LCR two-tower model.

Each sample:
    context_text  : str                 — citing passage (input to ContextTower)
    cited_paper_id: int                 — global integer ID of the cited paper
                                          (index into feat_P / feat_PP / feat_PCP tensors)

Source file: shared/data_prep/all_contexts.json
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
    - cited_uri must exist in node_index["paper"]  (already guaranteed by merge,
      but re-checked here for safety)
    - context must be non-empty after stripping

Split: deterministic random split seeded at 42
    train 80% / val 10% / test 10%
"""

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
    context_text:   str
    cited_paper_id: int   # global integer index into feat tensors
    cited_uri:      str   # kept for debugging / evaluation
    citing_uri:     str


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LCRDataset(Dataset): # list of pairs (context_text, cited_paper_id) 
    """
    Parameters
    ----------
    records : list[CitationRecord]  
    tokenizer : transformers tokenizer
    max_length : int
        Maximum token length for SciBERT (hard cap 512).
    """

    def __init__( # store record "list of citation record" + tokenizer
        self,
        records: List[CitationRecord],
        tokenizer,
        max_length: int = 256, # SciBERT default is 521 but since my context is 
    ):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx): # Takes one CitationRecord, runs the tokenizer on context_text and returns a dict with input_ids, attention_mask, and cited_paper_id
        rec = self.records[idx] # called once per sample per epoch.

        enc = self.tokenizer(
            rec.context_text, # cotext text to be tokenized
            max_length=self.max_length, # max number of tokens to keep
            truncation=True, # truncate extra tokens if context_text is longer than max_length
            padding=False,          # collate_fn handles padding
            return_tensors=None,    # return plain lists; collate pads to batch max
        )

        return {
            "input_ids":      enc["input_ids"], # batch of token sequences
            "attention_mask": enc["attention_mask"], # batch of masks (indicates which token to focus on and which to ignore(padded tokens))
            "cited_paper_id": rec.cited_paper_id, # batch of labels
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def lcr_collate_fn(batch): 
    #prepare the data for the pytorch's DataLoader to handle variable-length sequences by padding them to the same length within a batch.
    # It receives a batch of variable-length sequences (context) from __getitem__, 
    # finds the longest sequence in the batch
    #  add zeros to sequences shorter than the longest sequence (to the attention mask as well as the input_ids)
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
        "cited_paper_id": torch.tensor(cited_ids,      dtype=torch.long),
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
) -> dict:
    """
    Loads all_contexts.json, filters, maps URIs → integer IDs,
    splits into train/val/test, and returns LCRDataset objects.

    Returns
    -------
    {
        "train": LCRDataset,
        "val":   LCRDataset,
        "test":  LCRDataset,
        "tokenizer": tokenizer,
        "n_papers":  int,        # total number of paper nodes (corpus + external)
    }
    """
    # --- Load node index ---> mapping for all papers from node_index.json by  {uri → int_id} 
    print(f"Loading node index from {node_index_path} ...")
    with open(node_index_path, encoding="utf-8") as f:
        node_index = json.load(f)
    paper_uri_to_id = node_index["paper"]   # {uri: int_id}
    n_papers = len(paper_uri_to_id)
    print(f"  {n_papers:,} paper nodes in KG")

    # --- Load and filter records ---> iterates every record and build a flat list of CitationRecord objects from all_contexts.json
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
        # skip records with missing context or missing cited uri
        if not context_text:
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

    # --- Deterministic shuffle + split ---> shuffle the the list of CitationRecord "seed"=42 then split to 80/10/10 train/val/test
    # ---> split by citing URI's 80/10/10
    citing_uris = list({r.citing_uri for r in records})
    rng = random.Random(seed)
    rng.shuffle(citing_uris)
    
    n_uris       = len(citing_uris)
    n_train_uris = int(n_uris * train_ratio)
    n_val_uris   = int(n_uris * val_ratio)
    
    train_uris = set(citing_uris[:n_train_uris])
    val_uris   = set(citing_uris[n_train_uris : n_train_uris + n_val_uris])
    test_uris  = set(citing_uris[n_train_uris + n_val_uris :])
    
    train_records = [r for r in records if r.citing_uri in train_uris]
    val_records   = [r for r in records if r.citing_uri in val_uris]
    test_records  = [r for r in records if r.citing_uri in test_uris]

    print(f"  Split → train {len(train_records):,} / val {len(val_records):,} / test {len(test_records):,}")

    # --- Tokenizer ---> load tokenizer and wrap each split train/val/test inside LCRDataset (a custom PyTorch Dataset class)
    print(f"Loading tokenizer ({tokenizer_name}) ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return {
        "train":     LCRDataset(train_records, tokenizer, max_length),
        "val":       LCRDataset(val_records,   tokenizer, max_length),
        "test":      LCRDataset(test_records,  tokenizer, max_length),
        "tokenizer": tokenizer,
        "n_papers":  n_papers,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader

    BASE = os.path.expanduser("~/HGNN/shared/data_prep")

    result = build_datasets(
        all_contexts_path = os.path.join(BASE, "all_contexts.json"),
        node_index_path   = os.path.join(BASE, "node_index.json"),
        max_length        = 256,
    )

    for split in ("train", "val", "test"):
        ds = result[split]
        print(f"\n{split}: {len(ds):,} samples")
        sample = ds[0]
        print(f"  input_ids length : {len(sample['input_ids'])}")
        print(f"  cited_paper_id   : {sample['cited_paper_id']}")

    # DataLoader with collate
    loader = DataLoader(
        result["train"],
        batch_size=8,
        shuffle=True,
        collate_fn=lcr_collate_fn,
    )
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  input_ids      : {batch['input_ids'].shape}")
    print(f"  attention_mask : {batch['attention_mask'].shape}")
    print(f"  cited_paper_id : {batch['cited_paper_id'].shape}")
    print(f"  cited_paper_id values: {batch['cited_paper_id'].tolist()}")
    print(f"\nTotal paper nodes: {result['n_papers']:,}")
    print("\nStep C complete.")