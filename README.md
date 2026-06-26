# 🔬 HGNN for Local Citation Recommendation

> A two-tower dense retrieval model for **Local Citation Recommendation (LCR)** using a Heterogeneous Knowledge Graph — pairing a SeHGNN-based paper encoder with a fine-tuned SciBERT context encoder, trained with symmetric InfoNCE loss.


## 📁 Project Structure

```
HGNN/
├── paper_tower/
│   └── model.py                  # PaperTower: SeHGNN metapath fusion encoder
│
├── context_tower/
│   └── model.py                  # ContextTower: SciBERT + projection head
│
├── shared/
│   └── data_prep/
│       ├── dataset.py            # LCRDataset, lcr_collate_fn, build_datasets()
│       ├── all_contexts.json     # Flat citation context records (gitignored)
│       ├── node_index.json       # {paper_uri → global integer ID}
│       ├── feat_P.pt             # Pre-propagated metapath features  (gitignored)
│       ├── feat_PP.pt
│       ├── feat_PCP.pt
│       └── corpus_ids.pt         # Global IDs of corpus papers (for ranking)
│
├── checkpoints/                  # Training runs (gitignored)
│   └── <run_id>/
│       ├── best_model.pt         # Best checkpoint (by val MRR)
│       └── history.json          # Per-epoch loss + metrics log
│
├── train.py                      # Main training script
├── merge_kg.py                   # Merges per-paper TTL files → merged-kg.ttl
├── merged-kg.ttl                 # Full knowledge graph (gitignored)
└── .gitignore
```
