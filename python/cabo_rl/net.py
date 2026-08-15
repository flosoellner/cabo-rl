"""The learning agent's network: a small shared trunk plus one output head
per decision type in rules.Policy. Every decision is learned here - the old
agent only learned 3 of ~8 decision types and hard-coded the rest as greedy
heuristics (which position to swap/peek/spy/blind-swap into)."""
from __future__ import annotations

import torch
import torch.nn as nn

from cabo_rl.features import MAX_HAND, feature_dim

HEAD_SIZES: dict[str, int] = {
    "cabo": 2,  # [wait, call]
    "draw_source": 2,  # [pile, discard]
    "place_or_discard": 2,  # [place, discard_for_power]
    "use_group_discard": 2,  # [no, yes] - use the best known duplicate group, if any
    "swap_target": MAX_HAND,  # which of my positions the drawn/taken card fills
    "peek_target": MAX_HAND,  # which of my own positions to peek
    "spy_target": MAX_HAND,  # which of the opponent's positions to spy
    "swap_blind_decide": 2,  # [decline, accept]
    "swap_blind_own": MAX_HAND,  # my position to offer in a blind swap
    "swap_blind_opp": MAX_HAND,  # opponent's position to target in a blind swap
}


class CaboNet(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim(), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, size) for name, size in HEAD_SIZES.items()})

    def forward(self, x: torch.Tensor, head: str) -> torch.Tensor:
        return self.heads[head](self.trunk(x))
