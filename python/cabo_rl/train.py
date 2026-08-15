"""Self-play training: one shared CaboNet plays both seats against itself,
learns from each round's real outcome. Monte Carlo returns (like the old
agent's agentLearn), not bootstrapped TD - rounds are short enough that this
is simple, correct, and easy to reason about, at some sample-efficiency cost
we can revisit later (that's the NFSP upgrade path).

Runs on the temporarily-downsized single-round game we agreed on: a smaller
deck (values 0-13, same range so every power band and kamikaze are still
possible, just 2 copies of each instead of the real game's 2-4), same hand
size and rules otherwise.
"""
from __future__ import annotations

import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from cabo_rl import rules as R
from cabo_rl.agent import NetPolicy
from cabo_rl.features import Features
from cabo_rl.net import CaboNet, GLOBAL_HEAD_SIZES, HEAD_SIZES, POSITION_HEADS

# Shrunk-but-structurally-faithful training deck: every value 0-13 still
# appears (so every power band and kamikaze remain reachable), just 2
# copies each instead of the real game's 2-4 - roughly half the cards.
TRAIN_CARD_VALUES = [v for v in range(14) for _ in range(2)]
TRAIN_HAND_SIZE = 4
DECK_SIZE = len(TRAIN_CARD_VALUES)


def reward_for(round_points: dict[R.Who, int], who: R.Who) -> float:
    return float(round_points[R.other(who)] - round_points[who])


def play_training_round(net_policy: NetPolicy, rng: random.Random, starting: R.Who) -> dict[R.Who, int]:
    """Mirrors rules.play_round's loop, but also tells net_policy when a
    turn is the post-Cabo mandatory final turn (context the network should
    see, and rules.Policy has no generic hook for)."""
    state = R.GameState(deck=R.Deck(), players={"human": R.new_player("You"), "agent": R.new_player("Agent")})
    net_policy.reset_round()
    R.deal_new_round(state, rng, hand_size=TRAIN_HAND_SIZE, card_values=TRAIN_CARD_VALUES)

    current = starting
    cabo_caller: R.Who | None = None
    is_final = False
    for _ in range(200):
        net_policy.set_final_turn(is_final)
        called = R.take_turn(state, rng, current, net_policy, is_final)
        if called:
            cabo_caller = current
            is_final = True
            current = R.other(current)
            continue
        if is_final:
            break
        current = R.other(current)

    return R.resolve_round(state, cabo_caller)


class ReplayBuffer:
    def __init__(self, capacity: int = 30_000):
        self.buffers: dict[str, deque] = {h: deque(maxlen=capacity) for h in HEAD_SIZES}

    def push(self, head: str, features: Features, action: int, target: float) -> None:
        self.buffers[head].append((features, action, target))

    def sample(self, head: str, batch_size: int):
        buf = self.buffers[head]
        if len(buf) < batch_size:
            return None
        batch = random.sample(buf, batch_size)
        flat = np.stack([b[0].flat for b in batch])
        actions = np.array([b[1] for b in batch])
        targets = np.array([b[2] for b in batch], dtype=np.float32)
        if head in GLOBAL_HEAD_SIZES:
            return flat, None, actions, targets
        side = POSITION_HEADS[head]
        position_values = np.stack([(b[0].own_values if side == "own" else b[0].opp_values) for b in batch])
        return flat, position_values, actions, targets


def train(
    episodes: int = 20_000,
    batch_size: int = 64,
    train_every: int = 4,
    lr: float = 1e-3,
    eps_start: float = 0.3,
    eps_end: float = 0.03,
    seed: int = 0,
    log_every: int = 2000,
) -> CaboNet:
    # MPS exists but loses to plain CPU here (measured: ~5.3s vs ~8.8s for
    # 1000 episodes) - the network and batches are tiny, so per-op GPU
    # dispatch overhead dominates any real compute win. Worth revisiting
    # once the network/batch sizes grow (e.g. a history transformer).
    device = "cpu"
    net = CaboNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    buffer = ReplayBuffer()
    policy = NetPolicy(net, TRAIN_CARD_VALUES, DECK_SIZE, device=device)
    rng = random.Random(seed)

    t0 = time.time()
    for ep in range(1, episodes + 1):
        policy.epsilon = eps_start + (eps_end - eps_start) * (ep / episodes)
        starting: R.Who = "human" if ep % 2 == 0 else "agent"
        round_points = play_training_round(policy, rng, starting)

        for who, feats, head, action in policy.pop_trajectory():
            buffer.push(head, feats, action, reward_for(round_points, who))

        if ep % train_every == 0:
            for head in HEAD_SIZES:
                batch = buffer.sample(head, batch_size)
                if batch is None:
                    continue
                flat, position_values, actions, targets = batch
                flat_t = torch.from_numpy(flat).to(device)
                actions_t = torch.from_numpy(actions).long().to(device)
                targets_t = torch.from_numpy(targets).to(device)
                ctx = net.context(flat_t)
                if head in GLOBAL_HEAD_SIZES:
                    out = net.forward_global(ctx, head)
                else:
                    pos_t = torch.from_numpy(position_values).to(device)
                    out = net.forward_position(ctx, head, pos_t)
                pred = out.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                loss = nn.functional.mse_loss(pred, targets_t)
                opt.zero_grad()
                loss.backward()
                opt.step()

        if ep % log_every == 0:
            elapsed = time.time() - t0
            print(f"episode {ep}/{episodes}  eps={policy.epsilon:.3f}  elapsed={elapsed:.1f}s")

    return net


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--out", type=str, default="checkpoints/cabo_net.pt")
    args = parser.parse_args()

    trained = train(episodes=args.episodes)
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(trained.state_dict(), args.out)
    print(f"saved to {args.out}")
