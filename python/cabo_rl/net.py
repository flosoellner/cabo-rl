"""The learning agent's network: a shared trunk for global context, plus
two kinds of output:

- GLOBAL_HEAD_SIZES: plain linear heads for decisions that aren't "pick a
  position" (call cabo, draw source, place-vs-discard, ...).
- POSITION_HEADS: "which position" decisions (swap target, peek/spy/blind-
  swap targets) use a single SHARED scorer applied independently to each
  candidate position's own feature block, producing one score per
  position. This is the fix for a real, measured bug: a first version used
  a plain Linear(hidden, 4) head for these, which is free to learn
  position-index-specific quirks instead of a genuine "how good is this
  card" rule - measured accuracy on "pick the position holding the worst
  card" was 76%/75% for positions 1-2 (which the game's own rule always
  starts known, so they're heavily represented in training) but only
  42%/19% for positions 3-4. Scoring each position with the *same* small
  network, instead of one head that treats index as meaningful, forces the
  rule to generalize across position identity by construction - matching
  the project's own design notes on permutation-invariant hand encoding,
  which the first version skipped for speed and paid for in exactly this
  way.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cabo_rl.features import MAX_HAND, NUM_VALUES, feature_dim

GLOBAL_HEAD_SIZES: dict[str, int] = {
    "cabo": 2,  # [wait, call]
    "draw_source": 2,  # [pile, discard]
    "place_or_discard": 2,  # [place, discard_for_power]
    "use_group_discard": 2,  # [no, yes] - use the best known duplicate group, if any
    "swap_blind_decide": 2,  # [decline, accept]
}

# name -> which per-position feature block it scores over ("own" or "opp")
POSITION_HEADS: dict[str, str] = {
    "swap_target": "own",  # which of my positions the drawn/taken card fills
    "peek_target": "own",  # which of my own positions to peek
    "spy_target": "opp",  # which of the opponent's positions to spy
    "swap_blind_own": "own",  # my position to offer in a blind swap
    "swap_blind_opp": "opp",  # opponent's position to target in a blind swap
}

HEAD_SIZES: dict[str, int] = {**GLOBAL_HEAD_SIZES, **{name: MAX_HAND for name in POSITION_HEADS}}


class CaboNet(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim(), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.global_heads = nn.ModuleDict({name: nn.Linear(hidden, size) for name, size in GLOBAL_HEAD_SIZES.items()})
        # One shared scorer for every position-type decision - see module
        # docstring. Separate small MLPs per head (own vs opp) since "how
        # good is it to swap this in" and "who should I spy on" are
        # different questions, but each is still index-agnostic within
        # itself.
        self.position_scorers = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden + NUM_VALUES, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )
                for name in POSITION_HEADS
            }
        )

    def context(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def forward_global(self, ctx: torch.Tensor, head: str) -> torch.Tensor:
        return self.global_heads[head](ctx)

    def forward_position(self, ctx: torch.Tensor, head: str, position_values: torch.Tensor) -> torch.Tensor:
        """ctx: (batch, hidden). position_values: (batch, MAX_HAND, NUM_VALUES).
        Returns (batch, MAX_HAND) scores, one per candidate position."""
        batch, n_pos, _ = position_values.shape
        ctx_expanded = ctx.unsqueeze(1).expand(batch, n_pos, ctx.shape[-1])
        joint = torch.cat([ctx_expanded, position_values], dim=-1)
        scores = self.position_scorers[head](joint)  # (batch, n_pos, 1)
        return scores.squeeze(-1)
