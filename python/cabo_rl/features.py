"""Turns a cabo_rl.rules.GameState into a fixed-size feature vector for a
neural net, from one player's point of view.

Key design choice (per the project's design notes): unknown cards are
represented as a full probability distribution over possible values, not
collapsed to a single mean - the old tabular agent's `estimateHandValue`
used a scalar average, which throws away exactly the information a real
agent needs (e.g. "this card is almost certainly a face card" looks very
different from "this card averages 6.5" even when the mean matches).

A player's knowledge of the OPPONENT's specific cards (via spying) isn't
stored in GameState itself (see rules.py's documented asymmetry/memory
note) - the acting policy has to remember it. `Memory` here is that
policy-owned scratchpad, reset once per round.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from cabo_rl import rules as R

NUM_VALUES = 14  # values 0..13
MAX_HAND = 4  # hand never exceeds its starting size, only shrinks


@dataclass
class Memory:
    """A single policy instance's private notes for one round: exact values
    it has learned about the opponent's hand via its own spy actions.
    Reset at the start of every round - nothing persists across rounds,
    matching the game's own "no free memory" design."""

    opp_values: dict[int, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.opp_values.clear()


def unseen_distribution(state: R.GameState, who: R.Who, memory: Memory, card_values: list[int]) -> np.ndarray:
    """Probability distribution over the 14 possible values for any single
    still-unknown card slot, given everything `who` currently knows (own
    known cards, discard pile, and whatever it has personally spied)."""
    counts = Counter(card_values)
    player = state.players[who]
    opp = state.players[R.other(who)]
    for v, known in zip(player.hand, player.self_known):
        if known:
            counts[v] -= 1
    for pos, v in memory.opp_values.items():
        if pos < len(opp.hand):
            counts[v] -= 1
    for v in state.deck.discard_pile:
        counts[v] -= 1
    dist = np.zeros(NUM_VALUES, dtype=np.float32)
    total = sum(c for c in counts.values() if c > 0)
    if total <= 0:
        return np.full(NUM_VALUES, 1.0 / NUM_VALUES, dtype=np.float32)
    for v, c in counts.items():
        if c > 0 and 0 <= v < NUM_VALUES:
            dist[v] = c / total
    return dist


def _one_hot(v: int) -> np.ndarray:
    out = np.zeros(NUM_VALUES, dtype=np.float32)
    if 0 <= v < NUM_VALUES:
        out[v] = 1.0
    return out


def _encode_hand(
    hand: list[int],
    known_flags: list[bool],
    belief: np.ndarray,
    remembered: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (values: MAX_HAND x NUM_VALUES, active_mask: MAX_HAND)."""
    values = np.zeros((MAX_HAND, NUM_VALUES), dtype=np.float32)
    mask = np.zeros(MAX_HAND, dtype=np.float32)
    for i in range(min(len(hand), MAX_HAND)):
        mask[i] = 1.0
        if remembered is not None and i in remembered:
            values[i] = _one_hot(remembered[i])
        elif known_flags[i]:
            values[i] = _one_hot(hand[i])
        else:
            values[i] = belief
    return values, mask


def feature_dim() -> int:
    # own values + own mask + opp values + opp mask + discard top + draw
    # pile size + discard pile size + is_final_turn
    return (MAX_HAND * NUM_VALUES) + MAX_HAND + (MAX_HAND * NUM_VALUES) + MAX_HAND + NUM_VALUES + 1 + 1 + 1


def encode_state(
    state: R.GameState,
    who: R.Who,
    memory: Memory,
    card_values: list[int],
    deck_size: int,
    is_final_turn: bool,
) -> np.ndarray:
    player = state.players[who]
    opp = state.players[R.other(who)]
    belief = unseen_distribution(state, who, memory, card_values)

    own_values, own_mask = _encode_hand(player.hand, player.self_known, belief)
    opp_known_flags = [False] * len(opp.hand)
    opp_values, opp_mask = _encode_hand(opp.hand, opp_known_flags, belief, remembered=memory.opp_values)

    discard_top = (
        _one_hot(state.deck.discard_pile[-1]) if state.deck.discard_pile else np.zeros(NUM_VALUES, dtype=np.float32)
    )
    draw_size = np.array([len(state.deck.draw_pile) / max(deck_size, 1)], dtype=np.float32)
    discard_size = np.array([len(state.deck.discard_pile) / max(deck_size, 1)], dtype=np.float32)
    final_flag = np.array([1.0 if is_final_turn else 0.0], dtype=np.float32)

    return np.concatenate(
        [
            own_values.flatten(),
            own_mask,
            opp_values.flatten(),
            opp_mask,
            discard_top,
            draw_size,
            discard_size,
            final_flag,
        ]
    )
