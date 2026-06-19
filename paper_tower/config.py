# =============================================================================
# paper_tower/config.py
# Hyperparameters for the paper tower.
#
# Values marked [SeHGNN default] come from Table 8 / Appendix of the paper.
# Values marked [LCR TODO] need to be decided for the LCR task.
# =============================================================================

PAPER_TOWER_CONFIG = {

    # -------------------------------------------------------------------------
    # Metapath settings
    # -------------------------------------------------------------------------

    # Which metapath keys to consume from the precomputed feat_dict.
    # These must match the keys produced by shared/graph_propagation/hg_propagate.py.
    # [LCR TODO] Fill in once your adjacency dict naming is finalised.
    "feat_keys": [
        "P",        # raw paper features (SciBERT abstract embedding, 768-dim)
        "PP",       # 1-hop citation neighbourhood mean
        "PPP",      # 2-hop citation neighbourhood mean
        "PCP",      # paper → conference/venue → paper co-venue metapath
        "PCPr",     # paper → conference/venue → paper (reverse direction)
        # add more metapaths here as needed
    ],

    # Target node type code (single character used as dict key in feat_dict)
    "tgt_type": "P",

    # -------------------------------------------------------------------------
    # Input feature dimension
    # -------------------------------------------------------------------------

    # Raw input dim per metapath channel.
    # SciBERT CLS embedding = 768. If you use a different encoder, update this.
    # [LCR TODO] Confirm once SciBERT encoding of abstracts is ready.
    "nfeat": 768,

    # -------------------------------------------------------------------------
    # Model architecture  [SeHGNN defaults from Appendix]
    # -------------------------------------------------------------------------

    # Hidden dimension throughout the model (LinearPerMetapath output, Transformer width)
    "hidden": 512,          # [SeHGNN default: 512]

    # Number of MLP layers in the per-metapath feature projection block
    "n_fp_layers": 2,       # [SeHGNN default: 2]

    # Output embedding dimension (replaces nclass in SeHGNN; shared space with context tower)
    # [LCR TODO] Must match context_tower's output dim for InfoNCE similarity to make sense.
    "embed_dim": 256,

    # Number of attention heads in the Transformer semantic fusion module
    "num_heads": 1,         # [SeHGNN default: 1]

    # Activation function for the Transformer module
    "act": "none",          # [SeHGNN default: 'none']

    # -------------------------------------------------------------------------
    # Regularisation
    # -------------------------------------------------------------------------

    "dropout": 0.5,         # [SeHGNN default: 0.5]
    "input_drop": 0.1,      # [SeHGNN default: 0.1]
    "att_drop": 0.0,        # [SeHGNN default: 0.0]

    # Whether to add a residual branch from raw target-node features to the output
    "residual": False,      # [SeHGNN default for ACM: False; DBLP/Freebase: True]
                            # [LCR TODO] Tune on your validation set

    # -------------------------------------------------------------------------
    # Training  (placeholder — training loop lives in paper_tower/train.py)
    # -------------------------------------------------------------------------

    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 256,      # [LCR note] corpus is ~1,900 papers; full-batch may be feasible
    "epochs": 200,
}