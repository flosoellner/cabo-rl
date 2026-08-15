"""Play against the trained CaboNet in the terminal, on the same shrunk
single-round game it was trained on (not the real 52-card game yet - that's
a later step once this is actually worth scaling up).

    python cabo_rl/play_cli.py --checkpoint checkpoints/cabo_net.pt
"""
from __future__ import annotations

import argparse
import random

import torch

from cabo_rl import rules as R
from cabo_rl.agent import NetPolicy
from cabo_rl.net import CaboNet
from cabo_rl.train import DECK_SIZE, TRAIN_CARD_VALUES, TRAIN_HAND_SIZE


def ask(prompt: str, valid_choices: list[str]) -> str:
    while True:
        choice = input(prompt + " > ").strip().lower()
        if choice in valid_choices:
            return choice
        print(f"Please enter one of: {', '.join(valid_choices)}")


def ask_position(prompt: str, count: int) -> int:
    while True:
        raw = input(prompt + " > ").strip()
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw) - 1
        print(f"Please enter a number from 1 to {count}.")


def show_hand(state: R.GameState) -> None:
    h = state.players["human"]
    cells = [f"[{i + 1}:{v if k else '?'}]" for i, (v, k) in enumerate(zip(h.hand, h.self_known))]
    print("Your hand:  " + "  ".join(cells))
    print(f"Agent hand: {len(state.players['agent'].hand)} card(s), face-down")
    top = state.deck.discard_pile[-1] if state.deck.discard_pile else None
    print(f"Discard top: {top if top is not None else '(empty)'}   Draw pile: {len(state.deck.draw_pile)} left")


def flush_log(state: R.GameState, seen: int) -> int:
    for msg in state.log[seen:]:
        print("  " + msg)
    return len(state.log)


class HumanCliPolicy:
    """rules.Policy implemented via terminal prompts."""

    def decide_cabo(self, state: R.GameState, who: R.Who) -> bool:
        show_hand(state)
        return ask("Call CABO? [y/n]", ["y", "n"]) == "y"

    def decide_draw_source(self, state: R.GameState, who: R.Who) -> str:
        top = state.deck.discard_pile[-1]
        return "discard" if ask(f"Draw from [1] face-down pile or [2] take the visible {top}", ["1", "2"]) == "2" else "pile"

    def decide_place_or_discard(self, state: R.GameState, who: R.Who, card: int) -> str:
        print(f"You drew a {card} (only you see this).")
        power_note = " (has a power if discarded)" if card in R.POWER_PEEK_OWN | R.POWER_SPY_OPP | R.POWER_SWAP_BLIND else ""
        return "discard" if ask(f"[a] Swap it into your hand   [b] Discard it face-up{power_note}", ["a", "b"]) == "b" else "place"

    def choose_discard_positions(self, state: R.GameState, who: R.Who, card: int) -> list[int]:
        hand = state.players[who].hand
        show_hand(state)
        raw = input(
            f"Which position(s) get replaced by the {card}? One number for a plain swap, "
            f"or comma-separated positions you believe currently match each other (e.g. 2,4) > "
        ).strip()
        try:
            positions = sorted({int(x) - 1 for x in raw.split(",")})
        except ValueError:
            print("Could not read that - defaulting to position 1.")
            return [0]
        if not positions or any(p < 0 or p >= len(hand) for p in positions):
            print("Invalid selection - defaulting to position 1.")
            return [0]
        return positions

    def choose_peek_position(self, state: R.GameState, who: R.Who) -> int:
        return ask_position("Peek power: which of your own positions?", len(state.players[who].hand))

    def choose_spy_position(self, state: R.GameState, who: R.Who) -> int:
        return ask_position("Spy power: which of the agent's positions?", len(state.players["agent"].hand))

    def decide_swap_blind(self, state: R.GameState, who: R.Who) -> bool:
        return ask("Swap power: blind-swap one of your cards with the agent's? [y/n]", ["y", "n"]) == "y"

    def choose_swap_blind_positions(self, state: R.GameState, who: R.Who) -> tuple[int, int]:
        hpos = ask_position("Which of YOUR positions to give away?", len(state.players[who].hand))
        apos = ask_position("Which of the AGENT's positions to take (blind)?", len(state.players["agent"].hand))
        return hpos, apos


def play_one_round(net_policy: NetPolicy, human_policy: HumanCliPolicy, rng: random.Random, starting: R.Who) -> dict[R.Who, int]:
    state = R.GameState(deck=R.Deck(), players={"human": R.new_player("You"), "agent": R.new_player("Agent")})
    net_policy.reset_round()
    R.deal_new_round(state, rng, hand_size=TRAIN_HAND_SIZE, card_values=TRAIN_CARD_VALUES)
    print(f"\n{'=' * 60}\n{state.log[-2]}\n{state.log[-1]}")
    seen_log = len(state.log)

    policies = {"human": human_policy, "agent": net_policy}
    current = starting
    cabo_caller: R.Who | None = None
    is_final = False
    for _ in range(200):
        net_policy.set_final_turn(is_final)
        if current == "agent":
            print("\n--- Agent's turn ---")
        called = R.take_turn(state, rng, current, policies[current], is_final)
        seen_log = flush_log(state, seen_log)
        if called:
            print(f"{current.upper()} CALLS CABO!" if current == "human" else "Agent calls CABO!")
            cabo_caller = current
            is_final = True
            current = R.other(current)
            continue
        if is_final:
            break
        current = R.other(current)

    points = R.resolve_round(state, cabo_caller)
    seen_log = flush_log(state, seen_log)
    print(f"\nRound result: you={points['human']}  agent={points['agent']}")
    if points["human"] < points["agent"]:
        print("You win this round!")
    elif points["human"] > points["agent"]:
        print("Agent wins this round.")
    else:
        print("Tie.")
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cabo_net.pt")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    net = CaboNet()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.eval()
    net_policy = NetPolicy(net, TRAIN_CARD_VALUES, DECK_SIZE, device="cpu")
    net_policy.training = False
    net_policy.epsilon = 0.0
    human_policy = HumanCliPolicy()
    rng = random.Random(args.seed)

    print("Cabo vs. the trained agent - shrunk single-round training variant (values 0-13, 2 copies each, 4-card hands).")
    print("This is NOT the real 52-card game yet - it's the small variant the net was actually trained on.")

    session = {"human": 0, "agent": 0, "ties": 0}
    starting: R.Who = "human"
    while True:
        points = play_one_round(net_policy, human_policy, rng, starting)
        if points["human"] < points["agent"]:
            session["human"] += 1
        elif points["human"] > points["agent"]:
            session["agent"] += 1
        else:
            session["ties"] += 1
        print(f"Session score: you {session['human']} - agent {session['agent']} (ties {session['ties']})")
        starting = "agent" if starting == "human" else "human"
        if ask("Play another round? [y/n]", ["y", "n"]) == "n":
            break


if __name__ == "__main__":
    main()
