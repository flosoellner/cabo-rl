"""NetPolicy: a rules.Policy implementation driven by CaboNet. Every
decision point is a real learned choice (epsilon-greedy over a masked head)
except "which known-duplicate group to discard, if I have one" - identified
deterministically (you always know for certain when your own known cards
match, no judgment call there), leaving only the yes/no "is it worth taking
right now" as the learned part. Everything else the old agent hard-coded
(swap target, peek/spy/blind-swap targets) is a full learned head here.

One NetPolicy instance can play BOTH self-play seats at once (shared
weights - standard self-play), keeping separate per-seat memory and
trajectories internally.
"""
from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch

from cabo_rl import rules as R
from cabo_rl.features import Memory, encode_state
from cabo_rl.net import CaboNet, HEAD_SIZES


def find_best_known_group(hand: list[int], self_known: list[bool]) -> list[int] | None:
    """The largest group of positions the player *knows for certain* share a
    value (no guessing involved - equality of known cards is a fact, not a
    belief). Ties broken toward the higher value (discarding it removes more
    from a future score comparison)."""
    groups: dict[int, list[int]] = {}
    for i, (v, known) in enumerate(zip(hand, self_known)):
        if known:
            groups.setdefault(v, []).append(i)
    candidates = [g for g in groups.values() if len(g) >= 2]
    if not candidates:
        return None
    candidates.sort(key=lambda g: (len(g), hand[g[0]]), reverse=True)
    return candidates[0]


class NetPolicy:
    def __init__(self, net: CaboNet, card_values: list[int], deck_size: int, device: str = "cpu"):
        self.net = net
        self.card_values = card_values
        self.deck_size = deck_size
        self.device = device
        self.epsilon = 0.2
        self.training = True
        self.memory: dict[R.Who, Memory] = {"human": Memory(), "agent": Memory()}
        self.trajectory: list[tuple[R.Who, np.ndarray, str, int]] = []
        self._is_final_turn = False

    def reset_round(self) -> None:
        self.memory["human"].reset()
        self.memory["agent"].reset()

    def pop_trajectory(self) -> list[tuple[R.Who, np.ndarray, str, int]]:
        out = self.trajectory
        self.trajectory = []
        return out

    def set_final_turn(self, is_final: bool) -> None:
        self._is_final_turn = is_final

    # -- core: encode, mask, epsilon-greedy pick, optionally record --------

    def _features(self, state: R.GameState, who: R.Who) -> np.ndarray:
        return encode_state(
            state, who, self.memory[who], self.card_values, self.deck_size, self._is_final_turn
        )

    def _act(self, state: R.GameState, who: R.Who, head: str, valid: list[int]) -> int:
        feats = self._features(state, who)
        with torch.no_grad():
            logits = self.net(torch.from_numpy(feats).unsqueeze(0).to(self.device), head).squeeze(0).cpu().numpy()
        mask = np.full(HEAD_SIZES[head], -np.inf, dtype=np.float32)
        mask[valid] = logits[valid]

        if self.training and random.random() < self.epsilon:
            action = random.choice(valid)
        else:
            action = int(np.argmax(mask))

        if self.training:
            self.trajectory.append((who, feats, head, action))
        return action

    # -- rules.Policy interface --------------------------------------------

    def decide_cabo(self, state: R.GameState, who: R.Who) -> bool:
        return self._act(state, who, "cabo", [0, 1]) == 1

    def decide_draw_source(self, state: R.GameState, who: R.Who) -> Literal["pile", "discard"]:
        return "discard" if self._act(state, who, "draw_source", [0, 1]) == 1 else "pile"

    def decide_place_or_discard(self, state: R.GameState, who: R.Who, card: int) -> Literal["place", "discard"]:
        return "discard" if self._act(state, who, "place_or_discard", [0, 1]) == 1 else "place"

    def choose_discard_positions(self, state: R.GameState, who: R.Who, card: int) -> list[int]:
        player = state.players[who]
        group = find_best_known_group(player.hand, player.self_known)
        if group is not None:
            if self._act(state, who, "use_group_discard", [0, 1]) == 1:
                return group
        hand_len = len(player.hand)
        pos = self._act(state, who, "swap_target", list(range(hand_len)))
        return [pos]

    def choose_peek_position(self, state: R.GameState, who: R.Who) -> int:
        hand_len = len(state.players[who].hand)
        return self._act(state, who, "peek_target", list(range(hand_len)))

    def choose_spy_position(self, state: R.GameState, who: R.Who) -> int:
        opp = state.players[R.other(who)]
        pos = self._act(state, who, "spy_target", list(range(len(opp.hand))))
        # Remember what we learn - GameState itself doesn't persist this
        # (see rules.py's documented asymmetry), the acting policy must.
        self.memory[who].opp_values[pos] = opp.hand[pos]
        return pos

    def decide_swap_blind(self, state: R.GameState, who: R.Who) -> bool:
        return self._act(state, who, "swap_blind_decide", [0, 1]) == 1

    def choose_swap_blind_positions(self, state: R.GameState, who: R.Who) -> tuple[int, int]:
        own_len = len(state.players[who].hand)
        opp_len = len(state.players[R.other(who)].hand)
        own_pos = self._act(state, who, "swap_blind_own", list(range(own_len)))
        opp_pos = self._act(state, who, "swap_blind_opp", list(range(opp_len)))
        # The opponent position we're about to swap away will hold a
        # different, unknown card afterward - any stale memory of it (from
        # an earlier spy) is no longer valid.
        self.memory[who].opp_values.pop(opp_pos, None)
        return own_pos, opp_pos
