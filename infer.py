"""
infer.py  —  LCR inference
Usage:
  python infer.py --checkpoint ~/HGNN/checkpoints/<run_id>/best_model.pt
  python infer.py --checkpoint ... --query "We follow [CIT] for argument mining."
  
  python infer.py \
  --checkpoint ~/HGNN/checkpoints/20260622_120325/best_model.pt \
  --query "In the context of Dung’s abstract argumentation"

  python infer.py \
  --checkpoint ~/HGNN/checkpoints/20260622_120325/best_model.pt \
  --ttl_path ~/HGNN/shared/data_prep/merged-kg.ttl \
  --query "Discourse relations often carry ambiguous functions, where a single relation can simultaneously serve as both an elaboration and an argumentative justification. Prior annotation efforts have shown low inter-annotator agreement for these cases, particularly when examples and specifications are used to both illustrate a point and support a claim. Understanding this dual functionality is critical for improving discourse parsing and argument mining systems"

  python infer.py   --checkpoint ~/HGNN/checkpoints/20260622_120325/best_model.pt   --ttl_path ~/HGNN/shared/data_prep/merged-kg.ttl   --query "we took a closer look at two types of relations for which inter-framework agreement is particularly poor, and investigate the factors affecting interpretation of these relations in more detail. The relations under investigation are PDTB’s INSTANTIATION and SPECIFICATION relations (32% and 14% agreement, respectively). These relations do not have many prototypical connectives and are therefore hard to identify"
"""

import os
import sys
import json
import argparse
import torch
from transformers import AutoTokenizer

# Try to import rdflib
try:
    from rdflib import Graph, Namespace
    from rdflib.namespace import DCTERMS
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False
    print("❌ rdflib not installed. Please run: pip install rdflib")
    sys.exit(1)

ROOT = os.path.expanduser("~/HGNN")
sys.path.insert(0, os.path.join(ROOT, "paper_tower"))
sys.path.insert(0, os.path.join(ROOT, "context_tower"))
sys.path.insert(0, os.path.join(ROOT, "shared", "data_prep"))

from paper_tower.model   import PaperTower
from context_tower.model import ContextTower


def load_ttl_titles(ttl_path):
    """Load paper titles from merged TTL file using rdflib"""
    print(f"🔄 Parsing large TTL file with rdflib: {ttl_path}")
    print("This may take a while for big files...")

    g = Graph()
    try:
        g.parse(ttl_path, format="turtle")
        print(f"✅ Successfully parsed {len(g):,} triples.")
    except Exception as e:
        print(f"❌ Failed to parse TTL: {e}")
        return {}

    uri_to_title = {}
    for subj, pred, obj in g.triples((None, DCTERMS.title, None)):
        uri = str(subj)
        title = str(obj)
        
        # Prefer English titles and longer ones
        if (hasattr(obj, 'language') and obj.language and obj.language != 'en'):
            continue
        if uri not in uri_to_title or len(title) > len(uri_to_title[uri]):
            uri_to_title[uri] = title

    print(f"✅ Loaded {len(uri_to_title):,} paper titles from TTL.")
    return uri_to_title


def load_models(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt.get("args", {})

    paper_tower = PaperTower(
        feat_keys=["P", "PP", "PCP"], nfeat=768,
        hidden=a.get("hidden", 512), embed_dim=a.get("embed_dim", 256),
        n_fp_layers=a.get("n_fp_layers", 2),
        dropout=a.get("dropout", 0.5), input_drop=a.get("input_drop", 0.1),
    ).to(device)
    paper_tower.load_state_dict(ckpt["paper_tower"])
    paper_tower.eval()

    context_tower = ContextTower(
        embed_dim=a.get("embed_dim", 256), dropout=a.get("input_drop", 0.1),
    ).to(device)
    context_tower.load_state_dict(ckpt["context_tower"])
    context_tower.eval()

    print(f"Loaded model (epoch {ckpt.get('epoch','?')}, val MRR={ckpt.get('val_mrr',0):.4f})")
    return paper_tower, context_tower


@torch.no_grad()
def build_index(paper_tower, feats, corpus_ids, device):
    print("Building corpus index...")
    ids = corpus_ids.tolist()
    embs = []
    for i in range(0, len(ids), 512):
        batch = {k: v[ids[i:i+512]].to(device) for k, v in feats.items()}
        embs.append(paper_tower(batch).cpu())
    return torch.cat(embs).to(device), ids


@torch.no_grad()
def recommend(query, context_tower, corpus_embs, corpus_ids_list,
              tokenizer, uri_to_meta, device, topk=10, max_length=256):
    enc = tokenizer(query, max_length=max_length, truncation=True,
                    padding="max_length", return_tensors="pt")
    ctx_emb = context_tower(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    sims = torch.matmul(ctx_emb, corpus_embs.T).squeeze(0)
    scores, indices = sims.topk(topk)

    results = []
    for rank, (idx, score) in enumerate(zip(indices.tolist(), scores.tolist()), 1):
        meta = uri_to_meta.get(corpus_ids_list[idx], {})
        results.append({
            "rank": rank,
            "score": round(score, 4),
            "uri": meta.get("uri", f"id:{corpus_ids_list[idx]}"),
            "title": meta.get("title", "(no title)"),
        })
    return results


def print_results(results, query):
    print(f"\n{'='*100}")
    print(f"Query: {query[:80]}")
    print(f"{'='*100}")
    for r in results:
        print(f"#{r['rank']:2d}  score={r['score']:.4f}")
        print(f"URI:   {r['uri']}")
        print(f"Title: {r['title']}")
        print("-" * 90)
    print()


def main():
    p = argparse.ArgumentParser(description="LCR Inference with TTL titles")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--data_root", default="~/HGNN/shared/data_prep", help="Data directory")
    p.add_argument("--ttl_path", required=True, help="Path to your merged .ttl file")
    p.add_argument("--query", default=None, help="Single query string")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    data_root = os.path.expanduser(args.data_root)

    # Load models
    paper_tower, context_tower = load_models(os.path.expanduser(args.checkpoint), device)
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    # Load features and corpus
    feats = {k: torch.load(os.path.join(data_root, f"feat_{k}.pt"), map_location="cpu")
             for k in ["P", "PP", "PCP"]}
    corpus_ids = torch.load(os.path.join(data_root, "corpus_ids.pt"), map_location="cpu")

    corpus_embs, corpus_ids_list = build_index(paper_tower, feats, corpus_ids, device)

    # Load node index
    with open(os.path.join(data_root, "node_index.json")) as f:
        node_index = json.load(f)
    id_to_uri = {int_id: uri for uri, int_id in node_index["paper"].items()}

    # Load titles from TTL using rdflib
    ttl_titles = load_ttl_titles(os.path.expanduser(args.ttl_path))

    # Fallback to abstracts.json if needed
    abstracts = {}
    abs_path = os.path.join(data_root, "abstracts.json")
    if os.path.exists(abs_path):
        with open(abs_path) as f:
            abstracts = json.load(f)

    # Build metadata
    uri_to_meta = {}
    for int_id, uri in id_to_uri.items():
        title = ttl_titles.get(uri)
        if not title:
            title = abstracts.get(uri, "(no title)")
            if isinstance(title, str):
                title = title.splitlines()[0][:300]
        uri_to_meta[int_id] = {"uri": uri, "title": title or "(no title)"}

    rec_kwargs = {
        "context_tower": context_tower,
        "corpus_embs": corpus_embs,
        "corpus_ids_list": corpus_ids_list,
        "tokenizer": tokenizer,
        "uri_to_meta": uri_to_meta,
        "device": device,
        "topk": args.topk,
    }

    if args.query:
        print_results(recommend(args.query, **rec_kwargs), args.query)
        return

    # Interactive mode
    print(f"\nLCR Interactive Mode (top-{args.topk}) — type :quit to exit\n")
    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query == ":quit":
            break
        print_results(recommend(query, **rec_kwargs), query)


if __name__ == "__main__":
    main()
