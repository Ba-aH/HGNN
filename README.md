# Local Citation Recommendation via Heterogeneous Graph Neural Networks and Contrastive Learning


<p align="center">
  <img src="shared/figures/two_tower_architecture.png" alt="Two-tower training architecture" width="650">
</p>

## Repository Structure

```
SeHGNN/
├── train_baseline.py            # Baseline (P-only) training entry point
├── train_domain_specific.py     # Full training entry point (ArgKG / ACL-200, configurable feature set)
├── evaluate.py                  # Recall@n / group-aware MRR evaluation
├── infer.py                     # Inference / candidate ranking for a given citation context
│
├── paper_tower/
│   └── model.py                 # Paper Module: per-metapath MLPs + transformer fusion + projection head
├── context_tower/
│   └── model.py                 # Context Module: SciBERT encoder + projection head
│
├── data_preprocessing/          # SKG construction pipeline (Flask app + n8n workflows + RML mapping)
│   ├── requirements.txt
│   ├── mapping.rml.ttl
│   ├── Pipeline 1 – Reference Cleaning + CiteKG Generation.json
│   ├── Pipeline 2 - Context Extraction.json
│   ├── Pipeline 3 - KG Creation.json
│   ├── routes/
│   └── services/
│
├── shared/
│   ├── data_prep/                # ArgKG: graph construction → feature propagation (step1 → step5b)
│   └── acl200_prep/              # ACL-200: same stage structure (step1 → step4b)
│
├── configs/                      # Experiment configs, checkpoints, and training reports
│   ├── make_report.py            # Generates HTML reports (Recall@10 / MRR) across configuration runs
│   ├── P/, PP/, PPCon/, P+PP/, P+PPCon/, P+PP+PPCon/
│   └── acl200/
│
└── figures/
```

## Requirements

- Python 3.10+
- Docker (only needed if regenerating the knowledge graph from raw data via RML Mapper)

## Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Ba-aH/HGNN.git
cd SciteKG

python -m venv venv
on linux: source venv/bin/activate    
on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note**: `requirements.txt` that lives under `data_preprocessing/`. is for the pipeline for data preprocessing using the provided n8n-pipelines create another specific python enviroment following for it the same instruction as above.

## Reproducing the Results

### 1. Get the data

Both `shared/data_prep/` (ArgKG) and `shared/acl200_prep/` (ACL-200) ship with zipped data files. Unzip each `.zip` file in place before running anything:

```bash
cd shared/data_prep
unzip merged-kg.zip
unzip all_contexts.zip

cd ../acl200_prep
unzip abstracts.zip
unzip all_contexts.zip
```

### 2. Build the graph and propagate features

For ArgKG:

```bash
cd shared/data_prep
python step1_build_graph.py
python step2_encode_corpus_papers.py
python step3_encode_external_papers.py
python step4_assemble_and_propagate.py
python step5a_encode_citations.py
python step5b_propagate_PCCon.py
python freeze_split.py        # freezes split_uris.json for reproducibility across runs
```

For ACL-200:

```bash
cd shared/acl200_prep
python step1_build_graph.py
python step2_encode_papers.py
python step3_propagate.py
python step4a_encode_citations.py
python step4b_propagate_PCCon.py
```

### 3. Train

Each experiment is defined by a `config.json` file under `configs/<feature_set>/...`. Point the training script at the config you want to reproduce:

```bash
python train_domain_specific.py --config configs/P+PP/MLP_500/exp_PP_MLP_hidden500\(run1\)/config.json
```

To reproduce a specific row of the results table, pick the matching config folder — e.g. `configs/P/`, `configs/PP/`, `configs/PPCon/`, `configs/P+PP/MLP_500/`, `configs/P+PPCon/`, `configs/P+PP+PPCon/`, or `configs/acl200/`. Checkpoints and training history (`best_model.pt`, `history.json`, `test_curve.json`) are written back into that same experiment folder.

### 4. Evaluate

```bash
python evaluate.py --checkpoint configs/P+PP/MLP_500/exp_PP_MLP_hidden500\(run1\)/best_model.pt --dataset argkg
```

This reports Recall@n and group-aware MRR on the frozen test split.

### 5. Generate the sweep report

```bash
python configs/make_report.py
```

Produces an HTML report summarizing Recall@10 / MRR across all experiment folders under `configs/`.

## Trained Models

Pretrained checkpoints for the best-performing configurations reported in the paper (ArgKG P+PP, ACL-200 P+PP+PC):

- **Trained models**: [link]
- **[link]**

## Rebuilding the KG from Raw Data

If you want to reconstruct the knowledge graph from raw scraped papers instead of using the preprocessed release:

```bash
cd data_preprocessing
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

This starts the preprocessing service (reference cleaning, context extraction, RML mapping via Docker). Import and run `Pipeline 1`, `Pipeline 2`, and `Pipeline 3` (n8n workflow JSON files in this folder) in order.


