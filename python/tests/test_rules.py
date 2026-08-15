"""Scenario tests for cabo_rl.rules, mirroring reference/cabo-web/test_engine.mjs
so the Python port is checked against the same behavior the TS tests lock in.
Scenarios tied to the old tabular Q-agent (agentTryMatchDrawn, agentDecideCabo)
are out of scope here - rules.py deliberately excludes that policy."""
import random

import pytest

from cabo_rl import rules as R


def fresh_state(seed: int = 0) -> R.GameState:
    return R.new_game_state(random.Random(seed))


# --- deck composition ---
def test_deck_composition():
    counts: dict[int, int] = {}
    for v in R.CARD_VALUES:
        counts[v] = counts.get(v, 0) + 1
    assert len(R.CARD_VALUES) == 52
    for v in range(1, 13):
        assert counts[v] == 4, f"value {v} should have 4 copies"
    assert counts[0] == 2
    assert counts[13] == 2


# --- perform_blind_swap sync ---
def test_blind_swap_sync():
    state = fresh_state()
    state.players["human"].hand = [1, 2, 3, 4]
    state.players["human"].self_known = [True, True, False, True]
    state.players["agent"].hand = [9, 8, 7, 6]
    state.players["agent"].self_known = [True, False, True, True]
    state.opp_known = [True, False, False, True]

    R.perform_blind_swap(state, 0, 2)
    assert state.players["human"].hand[0] == 7
    assert state.players["agent"].hand[2] == 1
    assert state.players["human"].self_known[0] is False
    assert state.players["agent"].self_known[2] is False
    assert state.opp_known[0] is False


# --- hand_pop / hand_append keep opp_known aligned ---
def test_hand_pop_append_sync():
    state = fresh_state()
    state.players["human"].hand = [1, 1, 5, 9]
    state.players["human"].self_known = [True, True, True, False]
    state.opp_known = [True, True, False, False]

    val = R.hand_pop(state, "human", 1)
    assert val == 1
    assert state.players["human"].hand == [1, 5, 9]
    assert state.players["human"].self_known == [True, True, False]
    assert state.opp_known == [True, False, False]

    R.hand_append(state, "human", 42, True)
    assert state.players["human"].hand == [1, 5, 9, 42]
    assert state.opp_known == [True, False, False, False]


# --- kamikaze scoring ---
def test_kamikaze_scoring():
    state = fresh_state()
    state.players["human"].hand = [12, 12, 13, 13]
    state.players["agent"].hand = [1, 2, 3, 4]
    rp = R.resolve_round(state, "human")
    assert rp == {"human": 0, "agent": 50}


# --- cabo penalty scoring ---
def test_cabo_penalty_scoring():
    state = fresh_state()
    state.players["human"].hand = [1, 2, 3, 4]  # sum 10
    state.players["agent"].hand = [1, 1, 1, 1]  # sum 4
    rp = R.resolve_round(state, "human")  # human loses (10>4) and called cabo -> +5
    assert rp == {"human": 15, "agent": 0}


# --- tie scoring ---
def test_tie_scoring():
    state = fresh_state()
    state.players["human"].hand = [5, 5]
    state.players["agent"].hand = [3, 7]
    rp = R.resolve_round(state, "agent")
    assert rp == {"human": 10, "agent": 0}


# --- place_drawn_card: the exact reported bug scenario - drew a 6, discard
#     two 7s that match each other (NOT the drawn 6) ---
def test_place_drawn_card_matches_each_other_not_drawn_card():
    state = fresh_state()
    state.players["human"].hand = [7, 7, 2, 9]
    state.players["human"].self_known = [True, True, True, True]
    state.opp_known = [False, False, False, False]
    result = R.place_drawn_card(state, "human", 6, [0, 1])
    assert result.success is True
    assert len(state.players["human"].hand) == 3
    assert 6 in state.players["human"].hand
    assert 7 not in state.players["human"].hand


# --- place_drawn_card: a single selected card always succeeds ---
def test_place_drawn_card_single_is_plain_swap():
    state = fresh_state()
    state.players["human"].hand = [1, 2, 3, 4]
    state.players["human"].self_known = [True, True, True, True]
    state.opp_known = [False, False, False, False]
    result = R.place_drawn_card(state, "human", 9, [2])
    assert result.success is True
    assert state.players["human"].hand == [1, 2, 9, 4]


# --- place_drawn_card: mismatched selection fails, drawn card discarded as penalty ---
def test_place_drawn_card_mismatch_fails_cleanly():
    state = fresh_state()
    state.players["human"].hand = [7, 9, 2, 1]
    before = list(state.players["human"].hand)
    result = R.place_drawn_card(state, "human", 6, [0, 1])  # 7 and 9 don't match each other
    assert result.success is False
    assert state.players["human"].hand == before
    assert state.deck.discard_pile[-1] == 6


# --- place_drawn_card: three-of-a-kind ---
def test_place_drawn_card_three_of_a_kind():
    state = fresh_state()
    state.players["human"].hand = [5, 5, 5, 8]
    state.players["human"].self_known = [True, True, True, True]
    state.opp_known = [False, False, False, False]
    result = R.place_drawn_card(state, "human", 1, [0, 1, 2])
    assert result.success is True
    assert len(state.players["human"].hand) == 2
    assert sorted(state.players["human"].hand) == [1, 8]


# --- score threshold logic ---
def test_score_threshold_logic():
    state = fresh_state()
    state.players["human"].total_score = 95
    w = R.apply_round_scores(state, {"human": 5, "agent": 3})
    assert state.players["human"].total_score == 50
    assert w is None

    state.players["agent"].total_score = 98
    w = R.apply_round_scores(state, {"human": 0, "agent": 4})
    assert state.players["agent"].total_score == 102
    assert w == "human"


# --- determine_next_starter ---
def test_determine_next_starter():
    assert R.determine_next_starter({"human": 0, "agent": 12}, "agent", "human") == "human"
    assert R.determine_next_starter({"human": 8, "agent": 0}, "human", "human") == "agent"
    assert R.determine_next_starter({"human": 0, "agent": 0}, "human", "agent") == "human"
    assert R.determine_next_starter({"human": 0, "agent": 0}, None, "human") == "agent"


# --- Cabo lockout: the responding player's mandatory final turn never even
#     asks whether to call Cabo, let alone lets it succeed ---
def test_cabo_lockout_final_turn_cannot_call_cabo():
    class AlwaysCallsCabo:
        def __init__(self):
            self.decide_cabo_calls = 0

        def decide_cabo(self, state, who):
            self.decide_cabo_calls += 1
            return True

        def decide_draw_source(self, state, who):
            return "pile"

        def decide_place_or_discard(self, state, who, card):
            return "place"

        def choose_discard_positions(self, state, who, card):
            return [0]

        def choose_peek_position(self, state, who):
            return 0

        def choose_spy_position(self, state, who):
            return 0

        def decide_swap_blind(self, state, who):
            return False

        def choose_swap_blind_positions(self, state, who):
            return (0, 0)

    rng = random.Random(1)
    state = fresh_state()
    R.deal_new_round(state, rng)
    policy = AlwaysCallsCabo()

    called = R.take_turn(state, rng, "agent", policy, final=True)
    assert called is False, "a final (post-lockout) turn must never register as a Cabo call"
    assert policy.decide_cabo_calls == 0, "decide_cabo must not even be consulted on the mandatory final turn"


# --- full multi-round simulation via play_round + RandomPolicy: no exceptions ---
def test_full_simulation_no_exceptions():
    rng = random.Random(42)
    state = fresh_state(seed=42)
    policies: dict[R.Who, R.Policy] = {
        "human": R.RandomPolicy(random.Random(1)),
        "agent": R.RandomPolicy(random.Random(2)),
    }
    starting: R.Who = "human"
    rounds = 0
    winner = None
    for _ in range(40):
        if winner is not None:
            break
        outcome = R.play_round(state, rng, policies, starting)
        winner = outcome.winner
        starting = outcome.next_starter
        rounds += 1
    assert rounds > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
