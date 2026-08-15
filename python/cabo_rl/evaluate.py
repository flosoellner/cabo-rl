"""Evaluate a trained CaboNet against RandomPolicy (and optionally another
CaboNet) on the same shrunk single-round game it was trained on."""
from __future__ import annotations

import random

from cabo_rl import rules as R
from cabo_rl.agent import NetPolicy
from cabo_rl.train import DECK_SIZE, TRAIN_CARD_VALUES, TRAIN_HAND_SIZE


def play_eval_round(policies: dict[R.Who, R.Policy], rng: random.Random, starting: R.Who) -> dict[R.Who, int]:
    state = R.GameState(deck=R.Deck(), players={"human": R.new_player("You"), "agent": R.new_player("Agent")})
    for p in policies.values():
        if hasattr(p, "reset_round"):
            p.reset_round()
    R.deal_new_round(state, rng, hand_size=TRAIN_HAND_SIZE, card_values=TRAIN_CARD_VALUES)

    current = starting
    cabo_caller: R.Who | None = None
    is_final = False
    for _ in range(200):
        for p in policies.values():
            if hasattr(p, "set_final_turn"):
                p.set_final_turn(is_final)
        called = R.take_turn(state, rng, current, policies[current], is_final)
        if called:
            cabo_caller = current
            is_final = True
            current = R.other(current)
            continue
        if is_final:
            break
        current = R.other(current)

    return R.resolve_round(state, cabo_caller)


def evaluate_vs_random(net, episodes: int = 2000, seed: int = 123) -> dict:
    device = next(net.parameters()).device
    net_policy = NetPolicy(net, TRAIN_CARD_VALUES, DECK_SIZE, device=str(device))
    net_policy.training = False
    net_policy.epsilon = 0.0
    rng = random.Random(seed)

    wins = 0
    losses = 0
    ties = 0
    net_points_total = 0
    opp_points_total = 0

    for i in range(episodes):
        random_policy = R.RandomPolicy(random.Random(seed + i + 1), cabo_prob=0.05)
        net_seat: R.Who = "agent" if i % 2 == 0 else "human"
        opp_seat: R.Who = R.other(net_seat)
        policies = {net_seat: net_policy, opp_seat: random_policy}
        starting = net_seat if i % 4 < 2 else opp_seat

        points = play_eval_round(policies, rng, starting)
        net_pts, opp_pts = points[net_seat], points[opp_seat]
        net_points_total += net_pts
        opp_points_total += opp_pts
        if net_pts < opp_pts:
            wins += 1
        elif net_pts > opp_pts:
            losses += 1
        else:
            ties += 1

    return {
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / episodes,
        "avg_net_points": net_points_total / episodes,
        "avg_opp_points": opp_points_total / episodes,
    }


if __name__ == "__main__":
    import argparse

    import torch

    from cabo_rl.net import CaboNet

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cabo_net.pt")
    parser.add_argument("--episodes", type=int, default=2000)
    args = parser.parse_args()

    net = CaboNet()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.eval()

    stats = evaluate_vs_random(net, episodes=args.episodes)
    print(f"vs RandomPolicy over {stats['episodes']} rounds:")
    print(f"  win rate: {stats['win_rate']:.1%}  (wins={stats['wins']} losses={stats['losses']} ties={stats['ties']})")
    print(f"  avg points when net plays: {stats['avg_net_points']:.2f}")
    print(f"  avg points random opponent scores: {stats['avg_opp_points']:.2f}")
