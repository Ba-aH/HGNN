# Context Module (`context_tower/`)

Encodes a citation context (the text surrounding a citation marker) into the same shared embedding space as the Paper Module, so the two can be compared directly during contrastive training.


## Training

SciBERT is fine-tuned at a very low learning rate (`lr_scibert`, preferrably equalt to `2e-6`) to gently adapt its pretrained representations, while the projection head trains at a higher rate (`lr_head`, preferrably equalt to `1e-3`). Both towers are optimized jointly with the InfoNCE contrastive loss.

