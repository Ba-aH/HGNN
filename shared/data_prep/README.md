The code builds a citation graph and creates features for papers in 4 steps:

Step 1: Splits papers into corpus papers (those with abstracts that cite others) and external papers (only cited, no abstracts). Saves their IDs.
Step 2: Encodes corpus papers using their abstracts with SciBERT.
Step 3: Encodes external papers using the citation context passages around them.
Step 4: Combines all features and propagates them via citation and co-citation graphs to create metapath features (feat_P, feat_PP, feat_PCP).

(baha_env) jovyan@jupyter-behantous:~/HGNN/shared/data_prep$ python freeze_split.py \
        --all_contexts all_contexts.json \
        --node_index   node_index.json \
        --out          split_uris.json \
        --seed 42 --train_ratio 0.8 --val_ratio 0.1
Loading node index from node_index.json ...
  26,014 paper nodes in KG
Loading contexts from all_contexts.json ...
  82,868 raw records
  82,868 records kept, 0 skipped
  Split (by citing_uri) → train 1,416 / val 177 / test 177

Wrote frozen split → split_uris.json
  train/val/test citing_uris: 1,416/177/177