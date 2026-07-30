"""Boosting-style ensemble of tiny SimpleGraphConv "weak learners" -- the user's own idea:
replace one conv layer with K very small conv layers whose outputs are summed, trained
sequentially (gradient-boosting style) rather than jointly end-to-end.

Reuses utils/simple_gnn.py's SimpleGraphConv unmodified as the weak learner: `SimpleGraphConv(
in_dim, out_dim=1)` maps features straight to the scalar target via `lin_self(x) +
lin_neigh(weighted_neighbor_mean)` -- about as small a "conv matrix" as exists, and its lack of
an activation makes stage outputs directly, linearly additive, matching how boosted trees' leaf
values are just summed. This file only defines the architecture; utils/train_boosted_gnn.py
implements the actual sequential/staged fitting procedure (residual computation, freezing
earlier stages, per-stage optimizer).

Kept entirely separate from utils/simple_gnn.py -- only imports from it, never modifies it.
"""

import torch
import torch.nn as nn

from utils.simple_gnn import SimpleGraphConv


class BoostedGNN(nn.Module):
    """bias + shrinkage * sum_k stage_k(x), each stage_k a tiny SimpleGraphConv.

    Stages are meant to be added one at a time via add_stage() and fit against the residual left
    by all earlier (already-frozen) stages -- see train_boosted_gnn.py's fit_stage(). forward()
    always sums whatever is in `self.stages` so far (or `stages[:up_to]` if given), so a
    partially-built ensemble is already a valid, usable model -- exactly what's needed mid-boosting
    to compute the current residual for the next stage.
    """

    def __init__(self, in_dim, out_dim=1, shrinkage=0.1):
        super().__init__()
        self.shrinkage = shrinkage
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.stages = nn.ModuleList()
        self._in_dim, self._out_dim = in_dim, out_dim

    def add_stage(self):
        stage = SimpleGraphConv(self._in_dim, self._out_dim)
        self.stages.append(stage)
        return stage

    def freeze_stage(self, index):
        for p in self.stages[index].parameters():
            p.requires_grad_(False)

    def forward(self, x, edge_index, edge_weight, up_to=None):
        out = self.bias.expand(x.size(0), -1)
        stages = self.stages if up_to is None else self.stages[:up_to]
        for stage in stages:
            out = out + self.shrinkage * stage(x, edge_index, edge_weight)
        return out
