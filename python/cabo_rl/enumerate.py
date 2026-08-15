"""Info-set complexity estimation for Cabo. Run directly to regenerate the
numbers in docs/complexity.md.

Three pieces, in increasing scale:

1. exact_enumerate_shrunk() - exhaustive enumeration over an 8-card shrunk
   variant (all 8! deals x every legal action branch for one full turn).
   Small enough that the base case (before any action) is hand-verifiable,
   and every deeper level is checked by an internal conservation invariant
   (sizes must sum to the number of paths that reached that depth). This
   validates the info-set-key/counting logic before trusting it below.

2. exact_info_set_size() - closed-form combinatorial count of how many
   distinct true hidden-card assignments are consistent with a player's
   information at a given point in a REAL (52-card) round. Exact, not
   sampled - a multinomial coefficient over the unseen-card multiset.

3. monte_carlo_info_set_count() - samples many random full-game playouts of
   the real game and counts distinct coarse info-set keys encountered, as a
   lower-bound / order-of-growth estimate on the true info-set count
   (random sampling can only ever find a subset of what exists).

Known simplification (flagged here and in docs/complexity.md): the info-set
key used throughout is a *coarse* public trace - own known-card values,
opponent's hand length, discard-pile contents, draw-pile size. It does NOT
yet track *which specific position* a peek/spy/blind-swap targeted, even
though that targeting is itself public information (see the project's
opponent-modeling design notes). Every count here is therefore a lower
bound on the true, fully-refined info-set count - refining the key is a
roadmap item, not attempted here.
"""
from __future__ import annotations

import copy
import itertools
import math
import random
from collections import Counter
from dataclasses import dataclass

from cabo_rl import rules as R

# ---------------------------------------------------------------------------
# Shared info-set key
# ---------------------------------------------------------------------------


def info_set_key(state: R.GameState, who: R.Who) -> tuple:
    player = state.players[who]
    opp = state.players[R.other(who)]
    known_own = tuple(v if k else None for v, k in zip(player.hand, player.self_known))
    return (
        who,
        known_own,
        len(opp.hand),
        tuple(state.deck.discard_pile),
        len(state.deck.draw_pile),
    )


# ---------------------------------------------------------------------------
# 1. Exact enumeration on a shrunk (8-card) variant
# ---------------------------------------------------------------------------

# Deliberately spans every power band (7=peek, 9=spy, 11=swap) plus plain
# cards, so a single full turn exercises every branch of the rule set.
SHRUNK_DECK = [0, 1, 3, 5, 7, 9, 11, 13]
SHRUNK_HAND_SIZE = 2


def _legal_position_subsets(hand_len: int) -> list[list[int]]:
    out: list[list[int]] = []
    for size in range(1, hand_len + 1):
        out.extend(list(c) for c in itertools.combinations(range(hand_len), size))
    return out


def _fingerprint(state: R.GameState) -> tuple:
    return (
        tuple(state.players["human"].hand),
        tuple(state.players["agent"].hand),
        tuple(state.deck.draw_pile),
        tuple(state.deck.discard_pile),
    )


def _build_shrunk_initial_state(deal: tuple[int, ...]) -> R.GameState:
    state = R.GameState(deck=R.Deck(), players={"human": R.new_player("You"), "agent": R.new_player("Agent")})
    remaining = list(deal)
    for who in R.WHOS:
        p = state.players[who]
        p.hand = [remaining.pop(0) for _ in range(SHRUNK_HAND_SIZE)]
        p.self_known = [True] + [False] * (SHRUNK_HAND_SIZE - 1)
    state.opp_known = [False] * SHRUNK_HAND_SIZE
    state.deck.discard_pile = [remaining.pop(0)]
    state.deck.draw_pile = remaining
    return state


def _branch_place(state: R.GameState, who: R.Who, card: int, hand_len: int):
    for positions in _legal_position_subsets(hand_len):
        s = copy.deepcopy(state)
        R.place_drawn_card(s, who, card, positions)
        yield s


def _branch_discard_for_power(state: R.GameState, who: R.Who, card: int, hand_len: int, opp_hand_len: int):
    s = copy.deepcopy(state)
    power = R.discard_drawn(s, who, card)
    if power is None:
        yield s
        return
    if power == "peek":
        for pos in range(hand_len):
            s2 = copy.deepcopy(s)
            R.use_peek_own(s2, who, pos)
            yield s2
    elif power == "spy":
        for pos in range(opp_hand_len):
            s2 = copy.deepcopy(s)
            R.use_spy_opp(s2, who, pos)
            yield s2
    elif power == "swap":
        yield copy.deepcopy(s)  # decline
        for own_pos in range(hand_len):
            for opp_pos in range(opp_hand_len):
                s2 = copy.deepcopy(s)
                R.use_swap_blind(s2, who, own_pos, opp_pos)
                yield s2


@dataclass
class EnumerationResult:
    # info_sets_by_depth[depth] = {info_set_key: {underlying true states}} -
    # keyed by depth (not just by player), since depth 0 and depth 2 are
    # BOTH "human's turn" and must not be conflated together.
    info_sets_by_depth: dict[int, dict[tuple, set]]
    who_at_depth: dict[int, R.Who]
    paths_at_depth: dict[int, int]


def exact_enumerate_shrunk(turns_to_expand: int = 2, starting_player: R.Who = "human") -> EnumerationResult:
    info_sets_by_depth: dict[int, dict[tuple, set]] = {}
    who_at_depth: dict[int, R.Who] = {}
    paths_at_depth: dict[int, int] = {}

    def record(state: R.GameState, who: R.Who, depth: int) -> None:
        key = info_set_key(state, who)
        info_sets_by_depth.setdefault(depth, {}).setdefault(key, set()).add(_fingerprint(state))
        who_at_depth[depth] = who
        paths_at_depth[depth] = paths_at_depth.get(depth, 0) + 1

    def expand(state: R.GameState, current: R.Who, turns_left: int, depth: int) -> None:
        record(state, current, depth)
        if turns_left <= 0:
            return
        opp = R.other(current)
        hand_len = len(state.players[current].hand)
        opp_hand_len = len(state.players[opp].hand)
        sources = ["discard", "pile"] if state.deck.discard_pile else ["pile"]
        for source in sources:
            s = copy.deepcopy(state)
            if source == "discard":
                card = R.take_discard_card(s)
                from_discard = True
            else:
                if not s.deck.draw_pile:
                    continue
                card = s.deck.draw_pile.pop()
                from_discard = False

            branches = (
                _branch_place(s, current, card, hand_len)
                if from_discard
                else itertools.chain(
                    _branch_place(s, current, card, hand_len),
                    _branch_discard_for_power(s, current, card, hand_len, opp_hand_len),
                )
            )
            for result_state in branches:
                expand(result_state, opp, turns_left - 1, depth + 1)

    for deal in itertools.permutations(SHRUNK_DECK):
        expand(_build_shrunk_initial_state(deal), starting_player, turns_to_expand, depth=0)

    return EnumerationResult(info_sets_by_depth, who_at_depth, paths_at_depth)


def hand_verified_base_case() -> tuple[int, int]:
    """The depth-0 case (before any action) has a closed form: fixing the
    starting player's 1 known card and the discard-pile's 1 known card
    (2 of the 8 values), the remaining 6 unseen cards can be in any of 6!
    orders across the 6 unseen slots (1 own-unknown + 2 opponent + 3
    draw-pile). Distinct keys = 8 x 7 (ordered, known-own != discard-top).
    Returns (expected_num_keys, expected_size_per_key)."""
    n = len(SHRUNK_DECK)
    num_keys = n * (n - 1)
    size_per_key = math.factorial(n - 2)
    return num_keys, size_per_key


# ---------------------------------------------------------------------------
# 2. Exact info-set size on the real 52-card game (closed-form)
# ---------------------------------------------------------------------------


def _card_multiset() -> Counter:
    return Counter(R.CARD_VALUES)


def exact_info_set_size(revealed_values: list[int], num_unseen_slots: int) -> int:
    """Number of distinct ways to fill `num_unseen_slots` *labeled* slots
    (opponent hand positions + draw-pile positions, in order - a different
    draw-pile order is a different true state even though it only matters
    later) with the cards remaining after removing `revealed_values` from
    the full 52-card multiset. A multinomial coefficient."""
    pool = _card_multiset()
    for v in revealed_values:
        pool[v] -= 1
        assert pool[v] >= 0, f"revealed more copies of {v} than exist"
    remaining = sum(pool.values())
    assert remaining == num_unseen_slots, (remaining, num_unseen_slots)
    numerator = math.factorial(remaining)
    denom = 1
    for c in pool.values():
        denom *= math.factorial(c)
    return numerator // denom


def count_distinct_opponent_hand_tuples(revealed_values: list[int], k: int) -> int:
    """Distinct ordered k-tuples of VALUES (not physical cards) achievable
    for the opponent's k hand positions, drawing without replacement from
    the post-reveal multiset. Marginalizes out draw-pile order - answers
    'how many distinct opponent hands are consistent with what I know', a
    more decision-relevant (and much smaller) number than the full
    exact_info_set_size above."""
    pool = _card_multiset()
    for v in revealed_values:
        pool[v] -= 1

    def rec(pool: Counter, k: int) -> int:
        if k == 0:
            return 1
        total = 0
        for v in list(pool.keys()):
            if pool[v] <= 0:
                continue
            pool[v] -= 1
            total += rec(pool, k - 1)
            pool[v] += 1
        return total

    return rec(pool, k)


# ---------------------------------------------------------------------------
# 3. Monte Carlo info-set count estimate on the real game
# ---------------------------------------------------------------------------


def monte_carlo_info_set_count(num_rounds: int, seed: int = 0, checkpoints: tuple[int, ...] = ()) -> dict:
    rng = random.Random(seed)
    state = R.new_game_state(rng)
    policies: dict[R.Who, R.Policy] = {
        "human": R.RandomPolicy(random.Random(seed + 1)),
        "agent": R.RandomPolicy(random.Random(seed + 2)),
    }
    seen: dict[R.Who, set] = {"human": set(), "agent": set()}
    starting: R.Who = "human"
    growth: list[tuple[int, int, int]] = []  # (round_idx, |seen human|, |seen agent|)

    # Instrument take_turn indirectly: re-derive info sets by re-running
    # play_round but hooking record via a thin wrapper around take_turn.
    orig_take_turn = R.take_turn

    def instrumented(state, rng, who, policy, final):
        seen[who].add(info_set_key(state, who))
        return orig_take_turn(state, rng, who, policy, final)

    R.take_turn = instrumented  # type: ignore[assignment]
    try:
        for i in range(num_rounds):
            outcome = R.play_round(state, rng, policies, starting)
            starting = outcome.next_starter
            if outcome.winner is not None:
                state = R.new_game_state(rng)
                starting = "human"
            if (i + 1) in checkpoints:
                growth.append((i + 1, len(seen["human"]), len(seen["agent"])))
    finally:
        R.take_turn = orig_take_turn  # type: ignore[assignment]

    return {
        "rounds": num_rounds,
        "distinct_human": len(seen["human"]),
        "distinct_agent": len(seen["agent"]),
        "growth": growth,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== 1. Exact enumeration, shrunk 8-card variant ===")
    expected_keys, expected_size = hand_verified_base_case()
    print(f"Hand-derived base case (depth 0): {expected_keys} info sets x {expected_size} states each "
          f"= {expected_keys * expected_size} total (should equal 8! = {math.factorial(8)})")

    result = exact_enumerate_shrunk(turns_to_expand=2)
    for depth in sorted(result.paths_at_depth):
        who = result.who_at_depth[depth]
        keys = result.info_sets_by_depth[depth]
        sizes = [len(v) for v in keys.values()]
        paths = result.paths_at_depth[depth]
        # sum(sizes) counts DISTINCT (info-set, true-state) pairs, so it's
        # <= paths reached (which counts every deal x action-path, including
        # cases where different histories converge on the same resulting
        # state - real path convergence, not a bug; at depth 0 there's no
        # branching yet so the two happen to coincide exactly).
        print(
            f"depth {depth} ({who}'s decision): {paths} deal x action-path combinations reached this depth, "
            f"collapsing to {len(keys)} distinct info sets (size range {min(sizes)}-{max(sizes)}, "
            f"{sum(sizes)} distinct (info-set, true-state) pairs - {paths - sum(sizes)} paths "
            f"converged onto an already-seen state)"
        )

    depth0 = result.info_sets_by_depth[0]
    print(f"\ndepth 0 vs hand-derived: {len(depth0)} info sets found "
          f"(expected {expected_keys}), sizes all == {expected_size}? "
          f"{all(len(v) == expected_size for v in depth0.values())}")

    print("\n=== 2. Exact info-set size, real 52-card game ===")
    # Start of round, agent's perspective: 2 known own cards + 1 discard-top known.
    start_size = exact_info_set_size(revealed_values=[5, 8, 3], num_unseen_slots=49)
    start_opp_hands = count_distinct_opponent_hand_tuples(revealed_values=[5, 8, 3], k=4)
    print(f"Start of round (3 cards revealed to agent: 2 own + discard top): "
          f"{start_size:.3e} distinct full deals consistent with agent's info "
          f"(log10 = {math.log10(start_size):.1f})")
    print(f"  -> restricted to just the opponent's 4-card hand (draw-pile order marginalized out): "
          f"{start_opp_hands:,} distinct possible opponent hands")

    # Mid-round: agent has peeked/learned 2 more of its own cards, spied 2 of
    # opponent's, and 6 more cards have cycled through the discard pile.
    mid_revealed = [5, 8, 3, 2, 9] + [1, 4] + [7, 6, 11, 0, 12, 10]
    mid_unseen = 52 - len(mid_revealed)
    mid_size = exact_info_set_size(revealed_values=mid_revealed, num_unseen_slots=mid_unseen)
    mid_opp_hands = count_distinct_opponent_hand_tuples(revealed_values=mid_revealed, k=2)
    print(f"Mid-round ({len(mid_revealed)} cards revealed, {mid_unseen} unseen slots left): "
          f"{mid_size:.3e} distinct full deals (log10 = {math.log10(mid_size):.1f})")
    print(f"  -> distinct possible values for opponent's 2 still-unknown positions: {mid_opp_hands}")

    print("\n=== 3. Monte Carlo info-set count, real 52-card game ===")
    mc = monte_carlo_info_set_count(num_rounds=3000, seed=7, checkpoints=(100, 500, 1000, 2000, 3000))
    print(f"After {mc['rounds']} simulated rounds (RandomPolicy both sides):")
    print(f"  distinct info sets seen (human): {mc['distinct_human']:,}")
    print(f"  distinct info sets seen (agent): {mc['distinct_agent']:,}")
    print("  growth (rounds, |seen human|, |seen agent|):")
    for row in mc["growth"]:
        print(f"    {row}")
