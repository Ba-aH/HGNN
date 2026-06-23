# 🔬 HGNN for Local Citation Recommendation

> A two-tower dense retrieval model for **Local Citation Recommendation (LCR)** using a Heterogeneous Knowledge Graph — pairing a SeHGNN-based paper encoder with a fine-tuned SciBERT context encoder, trained with symmetric InfoNCE loss.

---

## ✨ Key Features

- **Two-tower retrieval architecture** — decouples context encoding from paper encoding for efficient inference-time retrieval
- **SeHGNN-based PaperTower** — encodes papers via pre-propagated metapath features (`P`, `PP`, `PCP`) through independent per-metapath MLPs and a cross-metapath Transformer
- **SciBERT ContextTower** — fully fine-tuned SciBERT with a lightweight projection head; encodes citing passages into the shared embedding space
- **Symmetric InfoNCE loss** — contrastive loss over in-batch negatives in both context→paper and paper→context directions
- **AMP training** — mixed-precision with `torch.amp` and gradient clipping for stable fine-tuning
- **Full-corpus ranking at eval time** — encodes the entire corpus once per evaluation epoch; ranks all papers per query
- **Metrics**: Recall@1/5/10/20, MRR, nDCG@10
- **Early stopping** — patience-based, keyed on validation MRR
- **Heterogeneous KG construction** — built from per-paper RDF triples using CiTO, FaBiO, C4O, FOAF, PRO ontologies, merged into a single deduplicated `merged-kg.ttl`
- **Deterministic train/val/test splits** — 80/10/10, split by citing URI (no leakage), seeded at 42

---

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

---

## 🏗️ Model Architecture

### ContextTower (`context_tower/model.py`)

Encodes a **citing passage** into a shared embedding space.

```
Input: token IDs [B, seq_len]
  └─ SciBERT (allenai/scibert_scivocab_uncased)   [fully fine-tuned]
       └─ CLS token hidden state  [B, 768]
            └─ Dropout → Linear(768, embed_dim) → LayerNorm → L2-norm
Output: [B, embed_dim]  (unit sphere)
```

- All SciBERT weights are trainable with a low learning rate (`lr_scibert = 2e-6`)
- The projection head uses a higher learning rate (`lr_head = 1e-3`)

---

### PaperTower (`paper_tower/model.py`)

Encodes a **paper** from its pre-propagated heterogeneous graph features.

```
Input: feat_dict {key → Tensor[B, 768]}  for keys P, PP, PCP
  └─ Stack → [B, M=3, 768]
       └─ InputDrop → LinearPerMetapath MLP × n_fp_layers  [B, M, hidden]
            └─ Cross-metapath Transformer (self-attention)  [B, M, hidden]
                 └─ Flatten → Linear(M×hidden, hidden)      [B, hidden]
                      └─ Linear(hidden, embed_dim) → L2-norm
Output: [B, embed_dim]  (unit sphere)
```

Metapaths:
| Key | Meaning |
|-----|---------|
| `P` | Paper self-features (SciBERT abstract embedding) |
| `PP` | Paper → Paper (direct citation hop) |
| `PCP` | Paper → Citation context → Paper (context-mediated hop) |

Features are propagated **offline** before training — no adjacency matrix is needed at forward time.

---

### InfoNCE Loss

Symmetric contrastive loss over in-batch negatives:

```
logits = context_emb @ paper_emb.T / temperature    [B, B]
labels = arange(B)                                   (diagonal = positives)

loss = 0.5 × (CE(logits, labels) + CE(logitsᵀ, labels))
```

Default temperature: `0.07` (SimCLR/CLIP convention).

---

## 🚀 Training

### Prerequisites

```bash
conda activate baha_env   # or your environment with PyTorch + HuggingFace
```

Required files in `--data_root`:
- `all_contexts.json` — flat citation context records
- `node_index.json` — paper URI → global integer ID mapping
- `feat_P.pt`, `feat_PP.pt`, `feat_PCP.pt` — pre-propagated metapath tensors
- `corpus_ids.pt` — 1-D LongTensor of corpus paper IDs (used for ranking at eval)

---

### Basic Usage

```bash
python train.py \
  --data_root  ~/HGNN/shared/data_prep \
  --output_dir ~/HGNN/checkpoints \
  --epochs     20 \
  --batch_size 64 \
  --embed_dim  256
```

---

### Resuming Training

There is no built-in `--resume` flag. To resume from a checkpoint manually:

```python
ckpt = torch.load("checkpoints/<run_id>/best_model.pt")
paper_tower.load_state_dict(ckpt["paper_tower"])
context_tower.load_state_dict(ckpt["context_tower"])
optimizer.load_state_dict(ckpt["optimizer"])
start_epoch = ckpt["epoch"] + 1
```

Then modify `train.py` to start the epoch loop from `start_epoch`.

---

### Checkpoint Format

Each run creates a timestamped directory under `--output_dir`:

```
checkpoints/20260612_143021/
├── best_model.pt     # Saved on every new val-MRR best
└── history.json      # [{epoch, train_loss, Recall@K, MRR, nDCG@10}, ...]
```

`best_model.pt` contains:

```python
{
  "epoch":         int,
  "paper_tower":   state_dict,
  "context_tower": state_dict,
  "optimizer":     state_dict,
  "val_mrr":       float,
  "metrics":       dict,
  "args":          dict,     # full argparse namespace serialised
}
```

---

## ⚙️ Configuration

All hyperparameters are set via CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `~/HGNN/shared/data_prep` | Path to preprocessed data |
| `--output_dir` | `~/HGNN/checkpoints` | Where to save checkpoints |
| `--embed_dim` | `256` | Shared embedding dimension (both towers) |
| `--hidden` | `512` | PaperTower hidden dimension |
| `--n_fp_layers` | `2` | Number of LinearPerMetapath MLP layers |
| `--dropout` | `0.5` | Activation dropout (PaperTower) |
| `--input_drop` | `0.1` | Input dropout (PaperTower) / ContextTower dropout |
| `--temperature` | `0.07` | InfoNCE temperature |
| `--epochs` | `20` | Maximum training epochs |
| `--batch_size` | `256` | Training batch size |
| `--max_length` | `256` | Max SciBERT token length (hard cap: 512) |
| `--lr_scibert` | `2e-6` | Learning rate for SciBERT weights |
| `--lr_head` | `1e-3` | Learning rate for ContextTower projection head |
| `--lr_paper` | `1e-3` | Learning rate for PaperTower |
| `--patience` | `5` | Early stopping patience (epochs without val MRR gain) |
| `--eval_every` | `1` | Evaluate every N epochs |
| `--seed` | `42` | Global random seed |
| `--gpu` | `0` | CUDA device index |

> **Note on `lr_scibert`:** The default was lowered from `1e-5` to `2e-6` after AMP overflow caused NaN loss during training. Keep it at `2e-6` or below. Gradient clipping (`max_norm=1.0`) is applied every batch.

---

## 📊 Evaluation

Evaluation runs at the end of every epoch (configurable via `--eval_every`).

**Protocol:**
1. Encode all corpus papers once using PaperTower → `[N_corpus, embed_dim]`
2. For each validation context, compute cosine similarity against all corpus embeddings
3. Find the rank of the ground-truth cited paper
4. Aggregate over all valid queries (external-only citations are skipped)

**Metrics reported:**

| Metric | Description |
|---|---|
| Recall@1/5/10/20 | Fraction of queries where the cited paper appears in top-K |
| MRR | Mean Reciprocal Rank |
| nDCG@10 | Normalised Discounted Cumulative Gain at cut-off 10 |

The checkpoint is saved whenever validation MRR improves.

---

## 🗄️ Data Pipeline

### `all_contexts.json`

Flat list of citation records produced by the preprocessing pipeline:

```json
[
  {
    "context":     "Graph neural networks have been widely used [CITATION].",
    "cited_uri":   "https://citekg.org/resource/paper/<hash>",
    "citing_uri":  "https://citekg.org/resource/paper/<hash>",
    "citing_idx":  42
  }
]
```

### `node_index.json`

Maps every paper URI to a global integer ID used to index into the metapath feature tensors:

```json
{
  "paper": {
    "https://citekg.org/resource/paper/<hash>": 0,
    ...
  }
}
```

### Train/Val/Test Split

Records are split **by citing URI** (not by record), ensuring all contexts from one citing paper land in the same split. This prevents leakage of citing-paper identity across sets.

- Train: 80% of citing URIs
- Val: 10%
- Test: 10%
- Seed: 42

---

## 🧠 Knowledge Graph

The KG is built from per-paper JSON files via an n8n + Flask + RMLMapper pipeline, serialised as RDF Turtle using:

| Ontology | Usage |
|---|---|
| `cito:` | Citation intent and relationships |
| `c4o:` | Citation context (in-text passages) |
| `fabio:` | Bibliographic object types |
| `foaf:` | Author identity |
| `pro:` | Author roles (`pro:RoleInTime` reification) |
| `bibo:` | Bibliographic metadata |
| `citekg:` | Custom linking property (`hasCitationContext`) |

All per-paper `paper-kg.ttl` files are merged and deduplicated into `merged-kg.ttl` via `merge_kg.py`.
