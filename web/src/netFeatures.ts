// Turns a GameState into the same feature layout cabo_rl/features.py
// produces in Python, so the ONNX-exported net (trained there) sees
// matching inputs here. Always encodes from the AGENT's point of view -
// the human is driven by real UI clicks, not this network.
//
// One simplification vs. the Python port: no separate "memory" object for
// spied values is needed here. `state.agentBrain.oppKnown[i]` already
// tracks exactly "has the agent legitimately learned position i", and
// engine.ts's own agent code has always read `human.hand[i]` directly
// whenever that flag is true (see the original agentUseSwapBlind) - the
// hiding of hand values from the player was always a rendering-level
// convention in main.ts, not real data isolation, so there's nothing extra
// to track here that GameState doesn't already carry.
import * as E from "./engine.js";
import type { GameState, Who } from "./engine.js";

export const NUM_VALUES = 14; // values 0..13
export const MAX_HAND = 4;

export interface Features {
  flat: Float32Array;
  ownValues: Float32Array; // MAX_HAND x NUM_VALUES, flattened
  ownMask: Float32Array; // MAX_HAND
  oppValues: Float32Array; // MAX_HAND x NUM_VALUES, flattened
  oppMask: Float32Array; // MAX_HAND
}

export function featureDim(): number {
  // own values + own mask + opp values + opp mask + discard top + drawn
  // card (pending, one-hot or zero if N/A) + draw pile size + discard
  // pile size + is_final_turn - must match cabo_rl/features.py exactly.
  return MAX_HAND * NUM_VALUES + MAX_HAND + MAX_HAND * NUM_VALUES + MAX_HAND + NUM_VALUES + NUM_VALUES + 1 + 1 + 1;
}

function other(who: Who): Who {
  return who === "human" ? "agent" : "human";
}

export function unseenDistribution(state: GameState, who: Who): Float32Array {
  const counts = new Map<number, number>();
  for (const v of E.CARD_VALUES) counts.set(v, (counts.get(v) ?? 0) + 1);

  const player = state.players[who];
  player.hand.forEach((v, i) => {
    if (player.selfKnown[i]) counts.set(v, (counts.get(v) ?? 0) - 1);
  });

  if (who === "agent") {
    const human = state.players.human;
    state.agentBrain.oppKnown.forEach((known, i) => {
      if (known && i < human.hand.length) counts.set(human.hand[i], (counts.get(human.hand[i]) ?? 0) - 1);
    });
  }

  for (const v of state.deck.discardPile) counts.set(v, (counts.get(v) ?? 0) - 1);

  const dist = new Float32Array(NUM_VALUES);
  let total = 0;
  counts.forEach((c) => {
    if (c > 0) total += c;
  });
  if (total <= 0) {
    dist.fill(1 / NUM_VALUES);
    return dist;
  }
  counts.forEach((c, v) => {
    if (c > 0 && v >= 0 && v < NUM_VALUES) dist[v] = c / total;
  });
  return dist;
}

function oneHot(v: number): Float32Array {
  const out = new Float32Array(NUM_VALUES);
  if (v >= 0 && v < NUM_VALUES) out[v] = 1.0;
  return out;
}

// remembered(i) returns a known value for position i if the caller already
// legitimately knows it (own known-flag, or oppKnown for the opponent),
// else null - falls back to the shared belief distribution.
function encodeHand(
  hand: number[],
  belief: Float32Array,
  remembered: (i: number) => number | null
): { values: Float32Array; mask: Float32Array } {
  const values = new Float32Array(MAX_HAND * NUM_VALUES);
  const mask = new Float32Array(MAX_HAND);
  for (let i = 0; i < Math.min(hand.length, MAX_HAND); i++) {
    mask[i] = 1.0;
    const known = remembered(i);
    const slice = known !== null ? oneHot(known) : belief;
    values.set(slice, i * NUM_VALUES);
  }
  return { values, mask };
}

// drawnCard: the value of the card currently pending a decision (place-vs-
// discard, which position it fills) - undefined/null when no card is
// pending yet (decide_cabo, decide_draw_source). Must be threaded through
// for any decision that judges a specific card's value - see
// cabo_rl/features.py's encode_state docstring for why this matters (an
// earlier version silently omitted it and could not have gotten this
// right, by construction, regardless of training).
export function encodeState(
  state: GameState,
  who: Who,
  deckSize: number,
  isFinalTurn: boolean,
  drawnCard: number | null = null
): Features {
  const opp = state.players[other(who)];
  const player = state.players[who];
  const belief = unseenDistribution(state, who);

  const own = encodeHand(player.hand, belief, (i) => (player.selfKnown[i] ? player.hand[i] : null));
  const oppEnc = encodeHand(opp.hand, belief, (i) =>
    who === "agent" && state.agentBrain.oppKnown[i] ? opp.hand[i] : null
  );

  const discardTop =
    state.deck.discardPile.length > 0
      ? oneHot(state.deck.discardPile[state.deck.discardPile.length - 1])
      : new Float32Array(NUM_VALUES);
  const drawnEnc = drawnCard !== null ? oneHot(drawnCard) : new Float32Array(NUM_VALUES);
  const drawSize = state.deck.drawPile.length / Math.max(deckSize, 1);
  const discardSize = state.deck.discardPile.length / Math.max(deckSize, 1);
  const finalFlag = isFinalTurn ? 1.0 : 0.0;

  const flat = new Float32Array(featureDim());
  let o = 0;
  flat.set(own.values, o);
  o += own.values.length;
  flat.set(own.mask, o);
  o += own.mask.length;
  flat.set(oppEnc.values, o);
  o += oppEnc.values.length;
  flat.set(oppEnc.mask, o);
  o += oppEnc.mask.length;
  flat.set(discardTop, o);
  o += discardTop.length;
  flat.set(drawnEnc, o);
  o += drawnEnc.length;
  flat[o++] = drawSize;
  flat[o++] = discardSize;
  flat[o++] = finalFlag;

  return { flat, ownValues: own.values, ownMask: own.mask, oppValues: oppEnc.values, oppMask: oppEnc.mask };
}
