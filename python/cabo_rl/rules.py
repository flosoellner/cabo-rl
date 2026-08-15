"""Cabo game rules engine - single-instance, faithful Python port of
``reference/cabo-web/src/engine.ts`` (the current, rule-correct
implementation - NOT ``reference/cabo_cli.py``, which predates several rule
fixes: the multi-discard equality-against-each-other fix and the Cabo
lockout among them).

Scope: pure game mechanics + round/turn orchestration only. Deliberately
excludes engine.ts's tabular Q-learning agent (agentDecide*/agentUse*/
qGet/qChoose/agentLearn/bucket/estimateHandValue/unseenPool/...) - that was
one specific toy policy, not part of the game's rules, and isn't needed for
info-set enumeration (enumerate.py) or a future NFSP/Deep CFR/ReBeL agent,
which will bring its own policy against these same rules. See
docs/roadmap.md.

One deliberate deviation from engine.ts: functions here take an explicit
``random.Random`` instance rather than using a global RNG, so games are
reproducible from a seed - needed to validate enumerate.py's shrunk-variant
exact counts and, later, for seeded counterfactual replay ("what if I'd
taken action a' instead, same deck order").

Known asymmetry carried over from engine.ts, not yet fixed: ``GameState``
tracks ``opp_known`` (the agent's belief about which of the human's cards it
knows) but has no mirror for "human's belief about the agent's hand" - the
original design assumed a human remembers things themselves. Fine for now;
once both seats are RL agents (self-play), this needs generalizing to a
``belief: dict[Who, list[bool]]`` - tracked as a roadmap item, not fixed
here to keep this port a faithful 1:1 reference.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

Who = Literal["human", "agent"]
PowerKind = Optional[Literal["peek", "spy", "swap"]]
WHOS: tuple[Who, Who] = ("human", "agent")


def other(who: Who) -> Who:
    return "agent" if who == "human" else "human"


def _build_card_values() -> list[int]:
    # 1-12 four times each, plus 0 and 13 twice each (matches a standard
    # deck: numbers + J/Q, 2 red kings = 0, 2 black kings = 13). An early
    # draft of this omitted 11 and 12 entirely - already caught and fixed.
    vals: list[int] = []
    for _ in range(4):
        vals.extend(range(1, 13))
    for _ in range(2):
        vals.append(0)
        vals.append(13)
    return vals


CARD_VALUES: list[int] = _build_card_values()
assert len(CARD_VALUES) == 52

POWER_PEEK_OWN = frozenset({7, 8})
POWER_SPY_OPP = frozenset({9, 10})
POWER_SWAP_BLIND = frozenset({11, 12})

MAX_SCORE = 100
RESET_SCORE = 50
CABO_PENALTY = 5
KAMIKAZE_BONUS = 50


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Player:
    name: str
    hand: list[int] = field(default_factory=list)
    self_known: list[bool] = field(default_factory=list)
    total_score: int = 0


@dataclass
class Deck:
    draw_pile: list[int] = field(default_factory=list)
    discard_pile: list[int] = field(default_factory=list)


@dataclass
class GameState:
    deck: Deck
    players: dict[Who, Player]
    opp_known: list[bool] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


def new_player(name: str) -> Player:
    return Player(name=name)


def new_deck(rng: random.Random, card_values: list[int] = CARD_VALUES) -> Deck:
    draw_pile = list(card_values)
    rng.shuffle(draw_pile)
    return Deck(draw_pile=draw_pile, discard_pile=[])


def new_game_state(rng: random.Random) -> GameState:
    return GameState(
        deck=new_deck(rng),
        players={"human": new_player("You"), "agent": new_player("Agent")},
        opp_known=[],
        log=[],
    )


def log_msg(state: GameState, msg: str) -> None:
    state.log.append(msg)


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------


def deck_draw(state: GameState, rng: random.Random) -> int:
    if not state.deck.draw_pile:
        deck_reshuffle(state, rng)
    if not state.deck.draw_pile:
        return 5  # extreme edge case, mirrors engine.ts
    return state.deck.draw_pile.pop()


def deck_reshuffle(state: GameState, rng: random.Random) -> None:
    discard = state.deck.discard_pile
    if len(discard) <= 1:
        return
    top = discard[-1]
    rest = discard[:-1]
    rng.shuffle(rest)
    state.deck.draw_pile = rest
    state.deck.discard_pile = [top]
    log_msg(state, "The face-down pile ran out - the discard pile was reshuffled.")


# ---------------------------------------------------------------------------
# Hand bookkeeping (mirrors handPop/handAppend/handSet/performBlindSwap)
# ---------------------------------------------------------------------------


def hand_pop(state: GameState, who: Who, idx: int) -> int:
    player = state.players[who]
    val = player.hand.pop(idx)
    player.self_known.pop(idx)
    if who == "human":
        state.opp_known.pop(idx)
    return val


def hand_append(state: GameState, who: Who, val: int, known: bool) -> None:
    player = state.players[who]
    player.hand.append(val)
    player.self_known.append(known)
    if who == "human":
        state.opp_known.append(False)


def hand_set(state: GameState, who: Who, idx: int, val: int, known: bool) -> None:
    player = state.players[who]
    player.hand[idx] = val
    player.self_known[idx] = known
    if who == "human":
        state.opp_known[idx] = False


def perform_blind_swap(state: GameState, human_idx: int, agent_idx: int) -> None:
    human = state.players["human"]
    agent = state.players["agent"]
    human.hand[human_idx], agent.hand[agent_idx] = agent.hand[agent_idx], human.hand[human_idx]
    human.self_known[human_idx] = False
    agent.self_known[agent_idx] = False
    state.opp_known[human_idx] = False


# ---------------------------------------------------------------------------
# Turn actions - generalized to `who`. engine.ts duplicated these into
# human_*/agent_* call sites (one driven by UI clicks, one by the bot's own
# inline logic); the underlying rule is the same regardless of who's acting,
# so this collapses them to a single set of functions any policy can drive.
# ---------------------------------------------------------------------------


def take_discard_card(state: GameState) -> int:
    return state.deck.discard_pile.pop()


def place_card_into(state: GameState, who: Who, card: int, pos: int) -> None:
    player = state.players[who]
    old_card = player.hand[pos]
    hand_set(state, who, pos, card, True)
    state.deck.discard_pile.append(old_card)
    log_msg(state, f"{who} places it at position {pos + 1}, discarding its old card ({old_card}) face-up.")


@dataclass
class PlaceDrawnResult:
    success: bool
    positions: list[int]
    values: list[int]


def place_drawn_card(state: GameState, who: Who, card: int, positions: list[int]) -> PlaceDrawnResult:
    """Generalizes engine.ts's humanPlaceDrawnCard. Selecting exactly 1
    position is a plain swap, always succeeds. Selecting 2-4 requires those
    positions to currently hold mutually-equal values (NOT equal to `card` -
    the drawn card's own value is irrelevant to the check, a real bug we
    once had backwards). On success the selected cards are discarded and
    `card` fills the resulting single slot. On failure, `card` itself is
    discarded as a penalty (it already left the pile) and the hand is
    otherwise untouched."""
    player = state.players[who]
    values = [player.hand[p] for p in positions]

    if len(positions) == 1:
        place_card_into(state, who, card, positions[0])
        return PlaceDrawnResult(True, positions, values)

    all_equal = len(values) > 0 and all(v == values[0] for v in values)
    if all_equal and len(positions) >= 2:
        for idx in sorted(positions, reverse=True):
            discarded_val = hand_pop(state, who, idx)
            state.deck.discard_pile.append(discarded_val)
        hand_append(state, who, card, True)
        log_msg(
            state,
            f"Success! {who} discarded {len(positions)} matching {values[0]}s and placed the drawn {card} instead.",
        )
        return PlaceDrawnResult(True, positions, values)

    state.deck.discard_pile.append(card)
    log_msg(state, f"Failed swap! Those cards were actually {values} - not all equal. The drawn card is discarded.")
    return PlaceDrawnResult(False, positions, values)


def discard_drawn(state: GameState, who: Who, card: int) -> PowerKind:
    """Only legal for a card drawn face-down from the pile - a card taken
    from the visible discard pile must be placed via place_drawn_card."""
    state.deck.discard_pile.append(card)
    log_msg(state, f"{who} discards the {card} face-up.")
    if card in POWER_PEEK_OWN:
        return "peek"
    if card in POWER_SPY_OPP:
        return "spy"
    if card in POWER_SWAP_BLIND:
        return "swap"
    return None


def use_peek_own(state: GameState, who: Who, pos: int) -> int:
    player = state.players[who]
    val = player.hand[pos]
    player.self_known[pos] = True
    log_msg(state, f"{who}'s position {pos + 1} is a {val} (hidden from {other(who)}).")
    return val


def use_spy_opp(state: GameState, who: Who, pos: int) -> int:
    opp = state.players[other(who)]
    val = opp.hand[pos]
    if who == "agent":
        state.opp_known[pos] = True
    log_msg(state, f"{who} spies on {other(who)}'s position {pos + 1}: a {val} (hidden from {other(who)}).")
    return val


def use_swap_blind(state: GameState, who: Who, own_pos: int, opp_pos: int) -> None:
    if who == "human":
        perform_blind_swap(state, own_pos, opp_pos)
    else:
        perform_blind_swap(state, opp_pos, own_pos)
    log_msg(
        state,
        f"{who} blind-swaps their position {own_pos + 1} with {other(who)}'s position {opp_pos + 1}. Neither value revealed.",
    )


# ---------------------------------------------------------------------------
# Policy interface + turn/round orchestration
# ---------------------------------------------------------------------------


class Policy(Protocol):
    """What any player (human UI, old tabular bot, future NFSP/Deep CFR/ReBeL
    agent) must decide to drive a turn. Each method only sees `state` and
    `who` - it's on the policy implementation to only look at information
    that player is actually allowed to know."""

    def decide_cabo(self, state: GameState, who: Who) -> bool: ...
    def decide_draw_source(self, state: GameState, who: Who) -> Literal["pile", "discard"]: ...
    def decide_place_or_discard(self, state: GameState, who: Who, card: int) -> Literal["place", "discard"]: ...
    def choose_discard_positions(self, state: GameState, who: Who, card: int) -> list[int]: ...
    def choose_peek_position(self, state: GameState, who: Who) -> int: ...
    def choose_spy_position(self, state: GameState, who: Who) -> int: ...
    def decide_swap_blind(self, state: GameState, who: Who) -> bool: ...
    def choose_swap_blind_positions(self, state: GameState, who: Who) -> tuple[int, int]: ...


def take_turn(state: GameState, rng: random.Random, who: Who, policy: Policy, final: bool) -> bool:
    """Runs one full turn for `who`. Returns True iff this turn was a Cabo
    call (no card is drawn in that case - play_round handles the resulting
    single mandatory final turn for the other player, who cannot call Cabo
    themselves - the lockout rule)."""
    if not final and policy.decide_cabo(state, who):
        log_msg(state, f"{who} declares CABO!")
        return True

    discard_pile = state.deck.discard_pile
    source: Literal["pile", "discard"] = (
        policy.decide_draw_source(state, who) if discard_pile else "pile"
    )

    if source == "discard":
        card = take_discard_card(state)
        from_discard = True
        log_msg(state, f"{who} takes the face-up card.")
    else:
        card = deck_draw(state, rng)
        from_discard = False

    if from_discard:
        # A card taken from the (already public) discard pile has no power -
        # it must be placed via the generalized swap-or-fail mechanic.
        positions = policy.choose_discard_positions(state, who, card)
        place_drawn_card(state, who, card, positions)
        return False

    action = policy.decide_place_or_discard(state, who, card)
    if action == "place":
        positions = policy.choose_discard_positions(state, who, card)
        place_drawn_card(state, who, card, positions)
        return False

    power = discard_drawn(state, who, card)
    if power == "peek":
        pos = policy.choose_peek_position(state, who)
        use_peek_own(state, who, pos)
    elif power == "spy":
        pos = policy.choose_spy_position(state, who)
        use_spy_opp(state, who, pos)
    elif power == "swap":
        if policy.decide_swap_blind(state, who):
            own_pos, opp_pos = policy.choose_swap_blind_positions(state, who)
            use_swap_blind(state, who, own_pos, opp_pos)
    return False


def deal_new_round(
    state: GameState,
    rng: random.Random,
    hand_size: int = 4,
    card_values: list[int] = CARD_VALUES,
) -> None:
    """`hand_size`/`card_values` default to the real game (4 cards, full 52-
    card deck) - overriding them plays a smaller variant with the exact same
    rules, used to train/evaluate a learning agent faster before scaling up.
    Mirrors the real rule of only knowing your first 2 cards, generalized as
    "first min(2, hand_size) known" for hand sizes other than 4."""
    state.deck = new_deck(rng, card_values)
    known_prefix = min(2, hand_size)
    for who in WHOS:
        p = state.players[who]
        p.hand = [state.deck.draw_pile.pop() for _ in range(hand_size)]
        p.self_known = [True] * known_prefix + [False] * (hand_size - known_prefix)
    state.opp_known = [False] * hand_size
    top = state.deck.draw_pile.pop()
    state.deck.discard_pile.append(top)
    state.log = []
    h = state.players["human"]
    log_msg(state, f"New round. Your first two cards: position 1 = {h.hand[0]}, position 2 = {h.hand[1]}.")
    log_msg(state, f"The starting face-up card on the discard pile is a {top}.")


def is_kamikaze(hand: list[int]) -> bool:
    return sorted(hand) == [12, 12, 13, 13]


def resolve_round(state: GameState, cabo_caller: Optional[Who]) -> dict[Who, int]:
    human = state.players["human"]
    agent = state.players["agent"]
    human_sum = sum(human.hand)
    agent_sum = sum(agent.hand)

    log_msg(state, f"Your final hand: {human.hand} (sum = {human_sum})")
    log_msg(state, f"Agent final hand: {agent.hand} (sum = {agent_sum})")

    human_kamikaze = is_kamikaze(human.hand)
    agent_kamikaze = is_kamikaze(agent.hand)

    round_points: dict[Who, int] = {"human": 0, "agent": 0}

    if human_kamikaze or agent_kamikaze:
        if human_kamikaze:
            log_msg(state, "KAMIKAZE! Your hand was exactly 12,12,13,13 - the agent takes +50!")
            round_points["human"] = 0
            round_points["agent"] += KAMIKAZE_BONUS
        if agent_kamikaze:
            log_msg(state, "KAMIKAZE! The agent's hand was exactly 12,12,13,13 - you take +50!")
            round_points["agent"] = 0
            round_points["human"] += KAMIKAZE_BONUS
    elif human_sum < agent_sum:
        round_points["human"] = 0
        round_points["agent"] = agent_sum
        if cabo_caller == "agent":
            round_points["agent"] += CABO_PENALTY
    elif agent_sum < human_sum:
        round_points["agent"] = 0
        round_points["human"] = human_sum
        if cabo_caller == "human":
            round_points["human"] += CABO_PENALTY
    else:
        if cabo_caller == "human":
            round_points["human"] = 0
            round_points["agent"] = agent_sum
        else:
            round_points["agent"] = 0
            round_points["human"] = human_sum

    return round_points


def apply_round_scores(state: GameState, round_points: dict[Who, int]) -> Optional[Who]:
    for who in WHOS:
        state.players[who].total_score += round_points[who]
    for who in WHOS:
        if state.players[who].total_score == MAX_SCORE:
            state.players[who].total_score = RESET_SCORE
            log_msg(state, f"{'You hit' if who == 'human' else 'The agent hit'} exactly 100 - reduced to 50!")
    for who in WHOS:
        if state.players[who].total_score > MAX_SCORE:
            return other(who)
    return None


def determine_next_starter(round_points: dict[Who, int], cabo_caller: Optional[Who], previous_starter: Who) -> Who:
    zero_players = [w for w in WHOS if round_points[w] == 0]
    if len(zero_players) == 1:
        return zero_players[0]
    if len(zero_players) == 2 and cabo_caller:
        return cabo_caller
    return other(previous_starter)


@dataclass
class RoundOutcome:
    round_points: dict[Who, int]
    winner: Optional[Who]
    next_starter: Who
    turns_played: int


def play_round(
    state: GameState,
    rng: random.Random,
    policies: dict[Who, Policy],
    starting_player: Who,
    max_turns: int = 500,
) -> RoundOutcome:
    """Plays one full round to completion: deals, alternates turns until
    someone calls Cabo (triggering exactly one final mandatory turn for the
    other player, per the lockout rule), then resolves and scores it.
    `max_turns` is a safety valve against a pathological policy that never
    calls Cabo - not part of the actual rules."""
    deal_new_round(state, rng)
    current = starting_player
    cabo_caller: Optional[Who] = None
    is_final_turn = False
    turns = 0

    while turns < max_turns:
        turns += 1
        called = take_turn(state, rng, current, policies[current], is_final_turn)
        if called:
            cabo_caller = current
            is_final_turn = True
            current = other(current)
            continue
        if is_final_turn:
            break
        current = other(current)

    round_points = resolve_round(state, cabo_caller)
    winner = apply_round_scores(state, round_points)
    next_starter = determine_next_starter(round_points, cabo_caller, starting_player)
    return RoundOutcome(round_points, winner, next_starter, turns)


class RandomPolicy:
    """Legal-but-unstrategic reference policy: exercises every branch of the
    rules (including failed multi-discards) without trying to play well.
    Used by tests and by enumerate.py's Monte Carlo info-set sampling - not
    meant as an opponent to learn against."""

    def __init__(self, rng: random.Random, cabo_prob: float = 0.05):
        self.rng = rng
        self.cabo_prob = cabo_prob

    def decide_cabo(self, state: GameState, who: Who) -> bool:
        return self.rng.random() < self.cabo_prob

    def decide_draw_source(self, state: GameState, who: Who) -> Literal["pile", "discard"]:
        return self.rng.choice(["pile", "discard"])

    def decide_place_or_discard(self, state: GameState, who: Who, card: int) -> Literal["place", "discard"]:
        return self.rng.choice(["place", "discard"])

    def choose_discard_positions(self, state: GameState, who: Who, card: int) -> list[int]:
        hand_len = len(state.players[who].hand)
        if self.rng.random() < 0.25 and hand_len >= 2:
            k = self.rng.randint(2, hand_len)
            return self.rng.sample(range(hand_len), k)
        return [self.rng.randrange(hand_len)]

    def choose_peek_position(self, state: GameState, who: Who) -> int:
        return self.rng.randrange(len(state.players[who].hand))

    def choose_spy_position(self, state: GameState, who: Who) -> int:
        return self.rng.randrange(len(state.players[other(who)].hand))

    def decide_swap_blind(self, state: GameState, who: Who) -> bool:
        return self.rng.random() < 0.5

    def choose_swap_blind_positions(self, state: GameState, who: Who) -> tuple[int, int]:
        own_len = len(state.players[who].hand)
        opp_len = len(state.players[other(who)].hand)
        return self.rng.randrange(own_len), self.rng.randrange(opp_len)
