"""
shared/data_prep/group_aware_sampler.py
----------------------------------------
BatchSampler that guarantees no two records sharing the same
context_group_id ever land in the same batch.

Why: a multi-citation marker like [1,2,3] produces several records with the
identical context_text but different cited_paper_id. If two such "sibling"
records ended up in the same batch, ContextTower would produce the same
embedding for both rows, and standard single-positive InfoNCE
(F.cross_entropy over the diagonal, labels = arange(batch_size)) would then
score each sibling's true paper as a hard negative for the other — two
contradictory gradients pulling the same embedding in opposite directions
within one training step.

We want to stay strictly faithful to the per-record ground truth (cited_uri
is the only correct label for its row, nothing relaxed or treated as a
secondary positive), so the fix lives entirely in the sampler: keep sibling
records apart across batches, and let train.py use the plain single-positive
cross-entropy loss unmodified — it never needs to know group_ids exist.

Packing strategy (multi-round anti-affinity packing, not bin-packing):
    - Each epoch, shuffle the order of groups AND the order of records
      within each group (reseeded epoch-to-epoch via set_epoch()).
    - Build a batch by scanning groups in the (shuffled) round order and
      taking at most one record from each group that hasn't already
      contributed to the batch, until batch_size is reached or every group
      has been checked.
    - A record whose group already contributed to the current batch simply
      waits — it becomes eligible again as soon as the next batch starts.
    - Repeat until every record has been placed. A group of size c needs at
      least c distinct batches (it can only ever contribute one record per
      batch), so the total number of batches is driven by the largest
      group, not by len(dataset) / batch_size. There is no "oversized
      batch" case here (unlike a co-locating sampler) — a large group is
      simply spread thin across many batches, one record each.
    - Batch order and within-batch order are also shuffled, so composition
      varies epoch to epoch the same way shuffle=True would.
"""

import random
from collections import defaultdict

from torch.utils.data import Sampler


class GroupAwareBatchSampler(Sampler):
    """
    Parameters
    ----------
    group_ids : list[int]
        group_ids[i] == context_group_id of dataset record i (same order as
        the underlying Dataset / LCRDataset.records).
    batch_size : int
    seed : int
        Base seed. Actual shuffle seed each epoch is seed + epoch, so call
        set_epoch(epoch) before each epoch to vary batch composition while
        staying reproducible.
    drop_last : bool
        If True, drop a final undersized trailing batch.
    """

    def __init__(self, group_ids, batch_size, seed=42, drop_last=False):
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        groups = defaultdict(list)
        for idx, gid in enumerate(group_ids):
            groups[gid].append(idx)
        self.groups = list(groups.values())  # list of list[int]

        largest = max(len(g) for g in self.groups)
        if largest > 1:
            print(
                f"[GroupAwareBatchSampler] Largest context_group_id has "
                f"{largest} sibling record(s) — at least {largest} batches "
                f"are needed this epoch so no two of them ever share a batch."
            )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _build_batches(self):
        rng = random.Random(self.seed + self.epoch)

        # Fresh shuffled copies each epoch: order within a group, and the
        # order groups are scanned in when filling a batch.
        queues = [list(g) for g in self.groups]
        for q in queues:
            rng.shuffle(q)
        rng.shuffle(queues)

        batches = []
        remaining = sum(len(q) for q in queues)

        while remaining > 0:
            batch = []
            for q in queues:
                if not q:
                    continue
                batch.append(q.pop())
                remaining -= 1
                if len(batch) == self.batch_size:
                    break
            if not batch:
                # Safety net — shouldn't trigger since remaining > 0 implies
                # at least one non-empty queue.
                break
            batches.append(batch)

        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()

        rng.shuffle(batches)
        for b in batches:
            rng.shuffle(b)

        return batches

    def __iter__(self):
        return iter(self._build_batches())

    def __len__(self):
        # Anti-affinity packing isn't a fixed count, but len(loader) is
        # checked by some training loops (e.g. progress bars) — recompute
        # on demand.
        return len(self._build_batches())