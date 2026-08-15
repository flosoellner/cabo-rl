#!/usr/bin/env python3
"""
CABO - play against a self-improving learning agent
=====================================================

Rules implemented (as specified by the designer of this variant):

- Deck: values 1-12 x4 copies + value 0 x2 copies + value 13 x2 copies (52 cards,
  matching a standard playing-card deck: number cards + Jack(11)/Queen(12), plus
  2 red Kings = 0 and 2 black Kings = 13). NOTE: the very first sketch of this
  game only listed `4 * [1..10] + 2 * [0, 13]`, which silently drops the 11s and
  12s that the power rules (and the Kamikaze rule) depend on -> fixed below.
- Each player starts with 4 face-down cards and may look at their own position
  1 and 2 only.
- On your turn you either:
    (a) take the face-up discard card and blind-swap it into one of your own
        positions (old card goes face-up on the discard pile), no power, OR
    (b) draw the top face-down card, look at it privately, then either
          - swap it into one of your own positions (old card -> discard pile,
            no power), or
          - discard it face-up. If it's a 7/8 you may peek at one of your own
            cards. If it's a 9/10 you may spy one of the opponent's cards
            (value stays hidden from the opponent). If it's an 11/12 you may
            blind-swap any one of your cards with any one of the opponent's
            (neither side sees the swapped values).
- Instead of a turn you may call CABO. The opponent gets exactly one more
  regular turn (no cabo), then all cards are revealed and scored.
- If you hold 2+ cards of a value you currently know, you may discard all of
  them and draw a single replacement card (face-down, hidden, or the visible
  top of the discard pile) -> your hand permanently shrinks by (matches - 1)
  cards, which is pure upside since fewer cards = fewer points. If you
  misjudge and the values weren't actually equal, your turn is wasted (no
  discard happens) - this can never happen to the agent since it never
  misremembers a card it has flagged as "known".
- Scoring at reveal: lower hand-sum wins (0 points). Loser scores their sum.
  If the winner is NOT the cabo-caller, the (losing) caller gets +5 extra.
  On a tie, the cabo-caller scores 0 and the opponent scores their sum.
  Special case ("Kamikaze"): a final hand of exactly {12, 12, 13, 13} scores
  0 for its owner and gives the opponent +50.
- Scores accumulate across rounds. Hitting exactly 100 resets you to 50.
  Exceeding 100 ends the match - the OTHER player wins.

Learning agent
--------------
Cabo has hidden information and a huge action space, so instead of learning
literally everything from scratch (which is what a full self-play system like
a chess/Go engine does, at massive compute cost), this agent splits its
behaviour into two parts:

 1. A handful of *strategic* decisions (draw from the pile vs. take the
    discard, swap a drawn card in vs. burn it for its power, call Cabo or
    not) are learned with tabular Q-learning: a lookup table that maps a
    coarse description of the situation to a value for each option, updated
    after every round based on how well the agent actually did. It explores
    randomly some of the time (epsilon-greedy) so it keeps discovering
    better options instead of only repeating its first habits.
 2. The *mechanical* micro-decisions (which exact card to peek/spy/swap) use
    simple hand-written heuristics driven by the agent's own memory of what
    it has legitimately seen, since learning those from scratch would need
    far more games than a casual human vs. agent session will ever play.

The Q-table is saved to cabo_agent_brain.json next to this script, so the
agent keeps whatever it learned the next time you run the script.
"""

import json
import os
import random
from collections import Counter

# --------------------------------------------------------------------------
# Deck
# --------------------------------------------------------------------------

CARD_VALUES = 4 * list(range(1, 13)) + 2 * [0, 13]  # 52 cards total

POWER_PEEK_OWN = {7, 8}
POWER_SPY_OPP = {9, 10}
POWER_SWAP_BLIND = {11, 12}

MAX_SCORE = 100
RESET_SCORE = 50
CABO_PENALTY = 5
KAMIKAZE_BONUS = 50
KAMIKAZE_HAND = [12, 12, 13, 13]

BRAIN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cabo_agent_brain.json")

ALPHA = 0.3
EPSILON_BASE = 0.25
EPSILON_MIN = 0.05


class Deck:
    def __init__(self):
        self.draw_pile = list(CARD_VALUES)
        random.shuffle(self.draw_pile)
        self.discard_pile = []

    def draw(self):
        if not self.draw_pile:
            self.reshuffle()
        if not self.draw_pile:
            # Extreme edge case: nothing left anywhere. Should not really
            # happen with a 52 card deck in a 2-player game.
            return 5
        return self.draw_pile.pop()

    def reshuffle(self):
        # Rule: when the face-down pile runs out, everything on the face-up
        # (discard) pile EXCEPT its top card is shuffled and becomes the new
        # face-down pile; the single top card stays face-up.
        if len(self.discard_pile) <= 1:
            return
        top = self.discard_pile[-1]
        rest = self.discard_pile[:-1]
        random.shuffle(rest)
        self.draw_pile = rest
        self.discard_pile = [top]
        print("(The face-down pile ran out - the discard pile was reshuffled.)")


# --------------------------------------------------------------------------
# Players / agent memory
# --------------------------------------------------------------------------

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []        # list[int] ground-truth values, index = position
        self.self_known = []  # list[bool] parallel to hand: do THEY currently know this card?
        self.total_score = 0


class AgentBrain:
    def __init__(self):
        self.opp_known = []          # list[bool] parallel to the human's hand
        self.qtables = default_qtables()
        self.games_played = 0
        self.trajectory = []         # [(decision_name, state_key, action), ...] this round


def default_qtables():
    return {"cabo": {}, "draw_source": {}, "swap_or_discard": {}}


def load_brain():
    brain = AgentBrain()
    if os.path.exists(BRAIN_FILE):
        try:
            with open(BRAIN_FILE) as f:
                data = json.load(f)
            brain.qtables = data.get("qtables", default_qtables())
            brain.games_played = data.get("games_played", 0)
        except Exception:
            pass
    return brain


def save_brain(brain):
    try:
        with open(BRAIN_FILE, "w") as f:
            json.dump({"qtables": brain.qtables, "games_played": brain.games_played}, f)
    except Exception as e:
        print(f"(Could not save agent learning progress: {e})")


class GameState:
    def __init__(self):
        self.deck = Deck()
        self.players = {"human": Player("You"), "agent": Player("Agent")}
        self.agent_brain = load_brain()


# --------------------------------------------------------------------------
# Hand bookkeeping helpers (keep parallel lists in sync)
# --------------------------------------------------------------------------

def hand_pop(state, who, idx):
    player = state.players[who]
    val = player.hand.pop(idx)
    player.self_known.pop(idx)
    if who == "human":
        state.agent_brain.opp_known.pop(idx)
    return val


def hand_append(state, who, val, known):
    player = state.players[who]
    player.hand.append(val)
    player.self_known.append(known)
    if who == "human":
        state.agent_brain.opp_known.append(False)


def hand_set(state, who, idx, val, known):
    player = state.players[who]
    player.hand[idx] = val
    player.self_known[idx] = known
    if who == "human":
        state.agent_brain.opp_known[idx] = False


def perform_blind_swap(state, human_idx, agent_idx):
    human = state.players["human"]
    agent = state.players["agent"]
    human.hand[human_idx], agent.hand[agent_idx] = agent.hand[agent_idx], human.hand[human_idx]
    human.self_known[human_idx] = False
    agent.self_known[agent_idx] = False
    state.agent_brain.opp_known[human_idx] = False


# --------------------------------------------------------------------------
# Probability / value estimation (agent's honest belief state)
# --------------------------------------------------------------------------

def unseen_pool(state):
    """All card values whose current location the agent does NOT know:
    candidates for the draw pile, the agent's unknown positions and the
    opponent's unknown positions."""
    counter = Counter(CARD_VALUES)
    agent = state.players["agent"]
    human = state.players["human"]
    for i, known in enumerate(agent.self_known):
        if known:
            counter[agent.hand[i]] -= 1
    for i, known in enumerate(state.agent_brain.opp_known):
        if known and i < len(human.hand):
            counter[human.hand[i]] -= 1
    if state.deck.discard_pile:
        counter[state.deck.discard_pile[-1]] -= 1
    pool = []
    for val, cnt in counter.items():
        if cnt > 0:
            pool.extend([val] * cnt)
    return pool


def expected_value(pool):
    return sum(pool) / len(pool) if pool else 5.0


def estimate_hand_value(hand, known_flags, avg_unknown):
    total = 0.0
    for v, k in zip(hand, known_flags):
        total += v if k else avg_unknown
    return total


def bucket(value, edges):
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


# --------------------------------------------------------------------------
# Tabular Q-learning helpers
# --------------------------------------------------------------------------

def get_epsilon(brain):
    return max(EPSILON_MIN, EPSILON_BASE - 0.002 * brain.games_played)


def q_get(table, state_key, actions, defaults=None):
    if state_key not in table:
        table[state_key] = {a: (defaults[a] if defaults and a in defaults else 0.0) for a in actions}
    return table[state_key]


def q_choose(table, state_key, actions, epsilon, defaults=None):
    qvals = q_get(table, state_key, actions, defaults)
    if random.random() < epsilon:
        return random.choice(actions)
    best_val = max(qvals[a] for a in actions)
    best_actions = [a for a in actions if qvals[a] == best_val]
    return random.choice(best_actions)


def record(state, decision_name, state_key, action):
    state.agent_brain.trajectory.append((decision_name, state_key, action))


def agent_learn(state, reward):
    brain = state.agent_brain
    for decision_name, state_key, action in brain.trajectory:
        table = brain.qtables[decision_name]
        qvals = q_get(table, state_key, [action])
        qvals.setdefault(action, 0.0)
        qvals[action] += ALPHA * (reward - qvals[action])
    brain.trajectory = []
    brain.games_played += 1
    save_brain(brain)


# --------------------------------------------------------------------------
# Agent strategic decisions (Q-learned)
# --------------------------------------------------------------------------

def agent_decide_cabo(state):
    brain = state.agent_brain
    avg = expected_value(unseen_pool(state))
    agent = state.players["agent"]
    human = state.players["human"]
    own_est = estimate_hand_value(agent.hand, agent.self_known, avg)
    opp_est = estimate_hand_value(human.hand, brain.opp_known, avg)
    unknown_own = agent.self_known.count(False)
    deck_left_bucket = bucket(len(state.deck.draw_pile), [10, 25])
    state_key = str((bucket(own_est, [8, 16, 26]), bucket(opp_est, [8, 16, 26]), unknown_own, deck_left_bucket))
    actions = ["call", "wait"]
    # Heuristic prior (used only the first time a state is seen): calling is
    # only attractive when our estimated hand is clearly better than the
    # opponent's AND we don't have too many still-unknown cards of our own
    # that could secretly be bad. This keeps early behaviour sane while the
    # Q-table is still empty; real experience overwrites it as games are played.
    margin = opp_est - own_est - unknown_own * 1.5
    defaults = {"call": margin * 0.5, "wait": 0.0}
    action = q_choose(brain.qtables["cabo"], state_key, actions, get_epsilon(brain), defaults)
    record(state, "cabo", state_key, action)
    return action == "call"


def agent_decide_draw_source(state):
    brain = state.agent_brain
    top = state.deck.discard_pile[-1] if state.deck.discard_pile else None
    if top is None:
        return "pile"
    avg = expected_value(unseen_pool(state))
    agent = state.players["agent"]
    own_est = estimate_hand_value(agent.hand, agent.self_known, avg)
    discard_bucket = 0 if top <= 3 else (1 if top <= 7 else 2)
    unknown_own = agent.self_known.count(False)
    state_key = str((bucket(own_est, [8, 16, 26]), discard_bucket, unknown_own))
    actions = ["pile", "discard"]
    # Heuristic prior: a visibly low discard card is worth grabbing; a high
    # one is worth avoiding (better to gamble on a hidden card instead).
    defaults = {"pile": 0.0, "discard": (avg - top) * 0.6}
    action = q_choose(brain.qtables["draw_source"], state_key, actions, get_epsilon(brain), defaults)
    record(state, "draw_source", state_key, action)
    return action


def agent_decide_swap_or_discard(state, card):
    brain = state.agent_brain
    avg = expected_value(unseen_pool(state))
    agent = state.players["agent"]
    own_est = estimate_hand_value(agent.hand, agent.self_known, avg)
    card_bucket = 0 if card <= 3 else (1 if card <= 7 else 2)
    has_power = 1 if (card in POWER_PEEK_OWN or card in POWER_SPY_OPP or card in POWER_SWAP_BLIND) else 0
    unknown_own = agent.self_known.count(False)
    state_key = str((bucket(own_est, [8, 16, 26]), card_bucket, has_power, unknown_own))
    actions = ["swap", "discard"]
    # Heuristic prior: swapping is attractive when the drawn card beats our
    # worst-estimated slot; discarding is mildly attractive on its own when
    # the card has a useful power attached.
    own_avg_per_card = own_est / max(len(agent.hand), 1)
    defaults = {"swap": (own_avg_per_card - card) * 0.6, "discard": 0.4 if has_power else -0.4}
    action = q_choose(brain.qtables["swap_or_discard"], state_key, actions, get_epsilon(brain), defaults)
    record(state, "swap_or_discard", state_key, action)
    return action


# --------------------------------------------------------------------------
# Agent tactical / mechanical heuristics (perfect-memory driven, not learned)
# --------------------------------------------------------------------------

def agent_choose_swap_position(state, card):
    agent = state.players["agent"]
    avg = expected_value(unseen_pool(state))
    best_pos, best_gain = 0, -10 ** 9
    for i, (v, k) in enumerate(zip(agent.hand, agent.self_known)):
        ref = v if k else avg
        gain = ref - card
        if gain > best_gain:
            best_gain = gain
            best_pos = i
    return best_pos


def agent_swap_in(state, pos, card, from_discard):
    agent = state.players["agent"]
    old_card = agent.hand[pos]
    hand_set(state, "agent", pos, card, True)
    state.deck.discard_pile.append(old_card)
    print(f"Agent places it at its position {pos + 1}, discarding its old card ({old_card}) face-up.")


def agent_use_peek_own(state):
    agent = state.players["agent"]
    unknown_positions = [i for i, k in enumerate(agent.self_known) if not k]
    if not unknown_positions:
        print("Agent's peek power fizzles (it already knows all its own cards).")
        return
    pos = random.choice(unknown_positions)
    agent.self_known[pos] = True
    print(f"Agent peeks at its own position {pos + 1} (value hidden from you).")


def agent_use_spy_opp(state):
    human = state.players["human"]
    brain = state.agent_brain
    unknown_positions = [i for i in range(len(human.hand)) if not brain.opp_known[i]]
    if not unknown_positions:
        print("Agent's spy power fizzles (it already knows all of your cards).")
        return
    pos = random.choice(unknown_positions)
    brain.opp_known[pos] = True
    print(f"Agent spies on your position {pos + 1} (value hidden from you).")


def agent_use_swap_blind(state):
    agent = state.players["agent"]
    human = state.players["human"]
    brain = state.agent_brain
    avg = expected_value(unseen_pool(state))

    worst_agent_pos = max(
        range(len(agent.hand)),
        key=lambda i: agent.hand[i] if agent.self_known[i] else avg,
    )
    our_val = agent.hand[worst_agent_pos] if agent.self_known[worst_agent_pos] else avg

    known_human_positions = [i for i in range(len(human.hand)) if brain.opp_known[i]]
    if known_human_positions:
        best_human_pos = min(known_human_positions, key=lambda i: human.hand[i])
        if our_val > human.hand[best_human_pos]:
            perform_blind_swap(state, best_human_pos, worst_agent_pos)
            print(f"Agent blind-swaps its position {worst_agent_pos + 1} with your position {best_human_pos + 1}.")
            return

    if agent.self_known[worst_agent_pos] and agent.hand[worst_agent_pos] >= 8 and len(human.hand) > 0:
        target_pos = random.randrange(len(human.hand))
        perform_blind_swap(state, target_pos, worst_agent_pos)
        print(f"Agent blind-swaps its position {worst_agent_pos + 1} with your position {target_pos + 1}.")
        return

    print("Agent chooses not to use its swap power this time.")


def agent_try_pair_discard(state):
    agent = state.players["agent"]
    value_positions = {}
    for i, (v, k) in enumerate(zip(agent.hand, agent.self_known)):
        if k:
            value_positions.setdefault(v, []).append(i)
    for v, positions in value_positions.items():
        if len(positions) >= 2:
            for idx in sorted(positions, reverse=True):
                discarded_val = hand_pop(state, "agent", idx)
                state.deck.discard_pile.append(discarded_val)
            top = state.deck.discard_pile[-1] if state.deck.discard_pile else None
            avg = expected_value(unseen_pool(state))
            if top is not None and top < avg:
                newcard = state.deck.discard_pile.pop()
            else:
                newcard = state.deck.draw()
            hand_append(state, "agent", newcard, True)
            print(f"Agent discards its matching {v}s ({len(positions)} cards) and draws one replacement.")
            return True
    return False


# --------------------------------------------------------------------------
# Agent turn
# --------------------------------------------------------------------------

def agent_turn(state, final=False):
    agent_try_pair_discard(state)

    if not final:
        if agent_decide_cabo(state):
            print("Agent declares CABO!")
            return True

    top_discard = state.deck.discard_pile[-1] if state.deck.discard_pile else None
    source = agent_decide_draw_source(state)

    if source == "discard" and top_discard is not None:
        card = state.deck.discard_pile.pop()
        pos = agent_choose_swap_position(state, card)
        print("Agent takes the face-up card.")
        agent_swap_in(state, pos, card, from_discard=True)
    else:
        card = state.deck.draw()
        action = agent_decide_swap_or_discard(state, card)
        if action == "swap":
            pos = agent_choose_swap_position(state, card)
            print("Agent draws from the pile.")
            agent_swap_in(state, pos, card, from_discard=False)
        else:
            state.deck.discard_pile.append(card)
            print(f"Agent draws from the pile and discards a {card} face-up.")
            if card in POWER_PEEK_OWN:
                agent_use_peek_own(state)
            elif card in POWER_SPY_OPP:
                agent_use_spy_opp(state)
            elif card in POWER_SWAP_BLIND:
                agent_use_swap_blind(state)
    return False


# --------------------------------------------------------------------------
# Human turn (console I/O)
# --------------------------------------------------------------------------

def ask(prompt, valid_choices):
    while True:
        choice = input(prompt + " > ").strip()
        if choice in valid_choices:
            return choice
        print(f"Please enter one of: {', '.join(valid_choices)}")


def ask_position(prompt, count):
    while True:
        raw = input(prompt + " > ").strip()
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw) - 1
        print(f"Please enter a number from 1 to {count}.")


def show_human_hand(state):
    h = state.players["human"]
    cells = []
    for i, (v, k) in enumerate(zip(h.hand, h.self_known)):
        cells.append(f"[{i + 1}:{v if k else '?'}]")
    print("Your hand:   " + "  ".join(cells))
    print(f"Agent hand:  {len(state.players['agent'].hand)} card(s), all face-down")


def try_discard_pairs(state, who):
    player = state.players[who]
    if len(player.hand) < 2:
        print("You don't have enough cards left to try this.")
        return
    raw = input("Enter the positions you believe match, separated by commas (e.g. 2,4) > ").strip()
    try:
        positions = sorted({int(x) - 1 for x in raw.split(",")})
    except ValueError:
        print("Could not read that, turn wasted.")
        return
    if len(positions) < 2 or any(p < 0 or p >= len(player.hand) for p in positions):
        print("Invalid selection, turn wasted.")
        return
    values = [player.hand[p] for p in positions]
    if len(set(values)) == 1:
        for idx in sorted(positions, reverse=True):
            discarded_val = hand_pop(state, who, idx)
            state.deck.discard_pile.append(discarded_val)
        top = state.deck.discard_pile[-1] if state.deck.discard_pile else None
        if top is not None:
            src = ask(f"Draw the replacement from [1] face-down pile (hidden) or [2] the visible {top} on the discard pile", ["1", "2"])
        else:
            src = "1"
        if src == "2":
            newcard = state.deck.discard_pile.pop()
        else:
            newcard = state.deck.draw()
        hand_append(state, who, newcard, True)
        print(f"Success! You discarded {len(positions)} matching {values[0]}s and drew a replacement: {newcard}.")
    else:
        print(f"Mismatch! Those cards were actually {values} - not all equal. Turn wasted, no cards discarded.")


def human_turn(state, final=False):
    print(f"\n--- Your turn --- (face-down pile: {len(state.deck.draw_pile)} cards left)")
    show_human_hand(state)
    top_discard = state.deck.discard_pile[-1] if state.deck.discard_pile else None
    print(f"Face-up discard pile top card: {top_discard}")

    if final:
        options = ["1", "2", "4"]
        prompt = "[1] Draw from face-down pile  [2] Take face-up discard card  [4] Discard matching cards"
    else:
        options = ["1", "2", "3", "4"]
        prompt = "[1] Draw from face-down pile  [2] Take face-up discard card  [3] Declare CABO  [4] Discard matching cards"

    choice = ask("Choose action: " + prompt, options)

    if choice == "3":
        print("You call CABO! The agent gets one last regular turn, then hands are revealed.")
        return True

    if choice == "4":
        try_discard_pairs(state, "human")
        return False

    if choice == "2":
        if top_discard is None:
            print("There is no face-up card to take.")
            return human_turn(state, final)
        card = state.deck.discard_pile.pop()
        pos = ask_position("Which of your positions should it replace (1-4)?", len(state.players["human"].hand))
        old_card = state.players["human"].hand[pos]
        hand_set(state, "human", pos, card, True)
        state.deck.discard_pile.append(old_card)
        print(f"You place the {card} at position {pos + 1}; your old card ({old_card}) is now face-up on the discard pile.")
        return False

    if choice == "1":
        card = state.deck.draw()
        print(f"You drew a {card} (only you can see this).")
        sub = ask("[a] Swap it into your hand   [b] Discard it face-up" +
                  ("  (uses its power if it has one)" if card in POWER_PEEK_OWN | POWER_SPY_OPP | POWER_SWAP_BLIND else ""),
                  ["a", "b"])
        if sub == "a":
            pos = ask_position("Which of your positions should it replace (1-4)?", len(state.players["human"].hand))
            old_card = state.players["human"].hand[pos]
            hand_set(state, "human", pos, card, True)
            state.deck.discard_pile.append(old_card)
            print(f"You place the {card} at position {pos + 1}; your old card ({old_card}) is now face-up on the discard pile.")
        else:
            state.deck.discard_pile.append(card)
            print(f"You discard the {card} face-up.")
            if card in POWER_PEEK_OWN:
                pos = ask_position("Peek power: which of your own positions do you want to look at (1-4)?", len(state.players["human"].hand))
                val = state.players["human"].hand[pos]
                state.players["human"].self_known[pos] = True
                print(f"(Position {pos + 1} is a {val}.)")
            elif card in POWER_SPY_OPP:
                pos = ask_position("Spy power: which of the agent's positions do you want to look at?", len(state.players["agent"].hand))
                val = state.players["agent"].hand[pos]
                print(f"(The agent's position {pos + 1} is a {val}. Remember it - the game won't remind you!)")
            elif card in POWER_SWAP_BLIND:
                do_swap = ask("Swap power: do you want to blind-swap one of your cards with one of the agent's? [y/n]", ["y", "n"])
                if do_swap == "y":
                    hpos = ask_position("Which of YOUR positions to give away (1-4)?", len(state.players["human"].hand))
                    apos = ask_position("Which of the AGENT's positions to take (blind)?", len(state.players["agent"].hand))
                    perform_blind_swap(state, hpos, apos)
                    print(f"You blind-swap your position {hpos + 1} with the agent's position {apos + 1}. Neither value was revealed.")
        return False


# --------------------------------------------------------------------------
# Round flow / scoring
# --------------------------------------------------------------------------

def deal_new_round(state):
    state.deck = Deck()
    for who in ("human", "agent"):
        p = state.players[who]
        p.hand = [state.deck.draw_pile.pop() for _ in range(4)]
        p.self_known = [True, True, False, False]
    state.agent_brain.opp_known = [False, False, False, False]
    top = state.deck.draw_pile.pop()
    state.deck.discard_pile.append(top)
    print("\n================ NEW ROUND ================")
    h = state.players["human"]
    print(f"Your first two cards: position 1 = {h.hand[0]}, position 2 = {h.hand[1]}  (positions 3 & 4 are unknown to you)")
    print(f"The starting face-up card on the discard pile is a {top}.")


def play_round(state, starting_player):
    current = starting_player
    cabo_caller = None
    while True:
        if current == "human":
            called_cabo = human_turn(state)
        else:
            called_cabo = agent_turn(state)

        if called_cabo:
            cabo_caller = current
            other = "agent" if current == "human" else "human"
            print(f"\n--- Final turn for {'you' if other == 'human' else 'the agent'} before the reveal ---")
            if other == "human":
                human_turn(state, final=True)
            else:
                agent_turn(state, final=True)
            break

        current = "agent" if current == "human" else "human"

    round_points = resolve_round(state, cabo_caller)
    return round_points, cabo_caller


def resolve_round(state, cabo_caller):
    human = state.players["human"]
    agent = state.players["agent"]
    human_sum = sum(human.hand)
    agent_sum = sum(agent.hand)

    print("\n--- REVEAL ---")
    print(f"Your final hand:   {human.hand}  (sum = {human_sum})")
    print(f"Agent final hand:  {agent.hand}  (sum = {agent_sum})")

    def is_kamikaze(hand):
        return sorted(hand) == KAMIKAZE_HAND

    human_kamikaze = is_kamikaze(human.hand)
    agent_kamikaze = is_kamikaze(agent.hand)

    round_points = {"human": 0, "agent": 0}

    if human_kamikaze or agent_kamikaze:
        if human_kamikaze:
            print("KAMIKAZE! Your hand was exactly {12,12,13,13} - the agent takes +50!")
            round_points["human"] = 0
            round_points["agent"] += KAMIKAZE_BONUS
        if agent_kamikaze:
            print("KAMIKAZE! The agent's hand was exactly {12,12,13,13} - you take +50!")
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


def apply_round_scores(state, round_points):
    for who in ("human", "agent"):
        state.players[who].total_score += round_points[who]
    for who in ("human", "agent"):
        if state.players[who].total_score == MAX_SCORE:
            state.players[who].total_score = RESET_SCORE
            print(f"{'You hit' if who == 'human' else 'The agent hit'} exactly 100 - reduced to 50!")
    for who in ("human", "agent"):
        if state.players[who].total_score > MAX_SCORE:
            other = "agent" if who == "human" else "human"
            return other
    return None


def determine_next_starter(round_points, cabo_caller, previous_starter):
    """Rule: the player who won the round (scored 0 points for it) starts
    the next round. In the rare case both players ended on 0 (a 0-0 tie),
    the cabo-caller is treated as the winner of that tie; if there's
    somehow still no clear winner, just alternate as a fallback."""
    zero_players = [w for w in ("human", "agent") if round_points[w] == 0]
    if len(zero_players) == 1:
        return zero_players[0]
    if len(zero_players) == 2 and cabo_caller:
        return cabo_caller
    return "agent" if previous_starter == "human" else "human"


def print_scores(state):
    h = state.players["human"].total_score
    a = state.players["agent"].total_score
    print(f"\nOverall score -> You: {h}   Agent: {a}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("Welcome to CABO. First to push the other player's score over 100 loses.")
    state = GameState()
    starting = random.choice(["human", "agent"])

    while True:
        deal_new_round(state)
        print_scores(state)
        print(f"{'You go' if starting == 'human' else 'The agent goes'} first this round.")
        round_points, cabo_caller = play_round(state, starting)
        print(f"\nRound result -> You: +{round_points['human']}   Agent: +{round_points['agent']}")
        winner = apply_round_scores(state, round_points)
        agent_learn(state, reward=round_points["human"] - round_points["agent"])
        print_scores(state)

        next_starter = determine_next_starter(round_points, cabo_caller, starting)

        if winner:
            print("\n=============================")
            print(f"GAME OVER - {'YOU WIN!' if winner == 'human' else 'The agent wins.'}")
            print("=============================")
            break

        starting = next_starter
        print(f"({'You' if starting == 'human' else 'The agent'} won that round and will start the next one.)")
        again = ask("\nPlay another round? [y/n]", ["y", "n"])
        if again == "n":
            save_brain(state.agent_brain)
            print("Progress saved. See you next time!")
            break


if __name__ == "__main__":
    main()