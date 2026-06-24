The code builds a citation graph and creates features for papers in 4 steps:

Step 1: Splits papers into corpus papers (those with abstracts that cite others) and external papers (only cited, no abstracts). Saves their IDs.
Step 2: Encodes corpus papers using their abstracts with SciBERT.
Step 3: Encodes external papers using the citation context passages around them.
Step 4: Combines all features and propagates them via citation and co-citation graphs to create metapath features (feat_P, feat_PP, feat_PCP).