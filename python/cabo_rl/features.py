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
    # own values + own mask + opp values + opp mask + discard top + drawn
    # card (pending, one-hot or zero if N/A) + draw pile size + discard
    # pile size + is_final_turn
    return (MAX_HAND * NUM_VALUES) + MAX_HAND + (MAX_HAND * NUM_VALUES) + MAX_HAND + NUM_VALUES + NUM_VALUES + 1 + 1 + 1


@dataclass
class Features:
    """Bundles the flattened context vector (for the trunk/global heads)
    with the raw per-position blocks (for the position-scoring heads) so
    both can be computed once and reused - see net.py for why position
    decisions need the raw per-position values, not just the flat vector."""

    flat: np.ndarray
    own_values: np.ndarray  # (MAX_HAND, NUM_VALUES)
    own_mask: np.ndarray  # (MAX_HAND,)
    opp_values: np.ndarray  # (MAX_HAND, NUM_VALUES)
    opp_mask: np.ndarray  # (MAX_HAND,)


def encode_state(
    state: R.GameState,
    who: R.Who,
    memory: Memory,
    card_values: list[int],
    deck_size: int,
    is_final_turn: bool,
    drawn_card: int | None = None,
) -> Features:
    """`drawn_card`: the value of the card currently pending a decision
    (e.g. "place vs discard for power", "which position does it fill") -
    None when no card is pending yet (decide_cabo, decide_draw_source).
    This is NOT optional in spirit - decisions that need to judge a
    specific card's value cannot be made sensibly without it. An earlier
    version silently omitted this entirely (accepted `card` as a parameter
    in the calling Policy methods but never threaded it into the features),
    so e.g. "which position should this card replace" was being decided
    with no idea what the card even was - only caught via manual review,
    not by any test, since a fixed-value smoke probe happened to hide it."""
    player = state.players[who]
    opp = state.players[R.other(who)]
    belief = unseen_distribution(state, who, memory, card_values)

    own_values, own_mask = _encode_hand(player.hand, player.self_known, belief)
    opp_known_flags = [False] * len(opp.hand)
    opp_values, opp_mask = _encode_hand(opp.hand, opp_known_flags, belief, remembered=memory.opp_values)

    discard_top = (
        _one_hot(state.deck.discard_pile[-1]) if state.deck.discard_pile else np.zeros(NUM_VALUES, dtype=np.float32)
    )
    drawn_enc = _one_hot(drawn_card) if drawn_card is not None else np.zeros(NUM_VALUES, dtype=np.float32)
    draw_size = np.array([len(state.deck.draw_pile) / max(deck_size, 1)], dtype=np.float32)
    discard_size = np.array([len(state.deck.discard_pile) / max(deck_size, 1)], dtype=np.float32)
    final_flag = np.array([1.0 if is_final_turn else 0.0], dtype=np.float32)

    flat = np.concatenate(
        [
            own_values.flatten(),
            own_mask,
            opp_values.flatten(),
            opp_mask,
            discard_top,
            drawn_enc,
            draw_size,
            discard_size,
            final_flag,
        ]
    )
    return Features(flat=flat, own_values=own_values, own_mask=own_mask, opp_values=opp_values, opp_mask=opp_mask)
