// Runs the agent's turn using the ONNX-exported CaboNet instead of the old
// tabular Q-agent. Mirrors cabo_rl/agent.py's NetPolicy + train.py's turn
// loop, but async (onnxruntime-web's session.run() is Promise-based) and
// always greedy (no epsilon exploration - this is a frozen, pretrained
// checkpoint doing inference, not continuing to learn in the browser).
//
// No bundler in this project (plain tsc -> native ES modules), so
// onnxruntime-web can't be imported by bare specifier - the browser
// wouldn't know where to resolve "onnxruntime-web" from. Instead the
// wasm-only build (no webgl/webgpu/node backends needed for a network
// this small) is vendored into public/ort/ and imported by relative path,
// same static-site philosophy as everything else in this app.
import * as ort from "../public/ort/ort.wasm.min.mjs";
import * as E from "./engine.js";
import type { GameState, Who, PowerKind } from "./engine.js";
import { encodeState, featureDim, MAX_HAND, NUM_VALUES, type Features } from "./netFeatures.js";

// wasmPaths must be resolvable regardless of deployment subpath (GitHub
// Pages serves this at /cabo-rl/, not domain root) - resolve relative to
// this module's own URL rather than using an absolute "/..." path.
ort.env.wasm.wasmPaths = new URL("../public/ort/", import.meta.url).href;

// The vendored .mjs has no real types (see ort-vendor.d.ts) - `ort` itself
// is `any`, so `ort.InferenceSession` can't be used as a *type*. This
// alias just documents intent at the call sites below.
export type OrtSession = any;

// Must match cabo_rl/export_onnx.py's OUTPUT_NAMES order exactly.
const GLOBAL_HEADS = ["cabo", "draw_source", "place_or_discard", "use_group_discard", "swap_blind_decide"] as const;
const POSITION_HEADS: Record<string, "own" | "opp"> = {
  swap_target: "own",
  peek_target: "own",
  spy_target: "opp",
  swap_blind_own: "own",
  swap_blind_opp: "opp",
};
type HeadName = (typeof GLOBAL_HEADS)[number] | keyof typeof POSITION_HEADS;

let sessionPromise: Promise<OrtSession> | null = null;

export function loadModel(modelUrl: string): Promise<OrtSession> {
  const existing: Promise<OrtSession> | null = sessionPromise;
  if (existing) return existing;
  const created: Promise<OrtSession> = ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
  sessionPromise = created;
  return created;
}

function findBestKnownGroup(hand: number[], selfKnown: boolean[]): number[] | null {
  const groups = new Map<number, number[]>();
  hand.forEach((v, i) => {
    if (selfKnown[i]) {
      const arr = groups.get(v) ?? [];
      arr.push(i);
      groups.set(v, arr);
    }
  });
  const candidates = [...groups.values()].filter((g) => g.length >= 2);
  if (candidates.length === 0) return null;
  candidates.sort((a, b) => (b.length - a.length) || (hand[b[0]] - hand[a[0]]));
  return candidates[0];
}

async function runAllHeads(
  session: OrtSession,
  feats: Features
): Promise<Record<HeadName, Float32Array>> {
  const feeds = {
    flat: new ort.Tensor("float32", feats.flat, [1, featureDim()]),
    own_values: new ort.Tensor("float32", feats.ownValues, [1, MAX_HAND, NUM_VALUES]),
    opp_values: new ort.Tensor("float32", feats.oppValues, [1, MAX_HAND, NUM_VALUES]),
  };
  const results = await session.run(feeds);
  const out = {} as Record<HeadName, Float32Array>;
  for (const name of [...GLOBAL_HEADS, ...Object.keys(POSITION_HEADS)] as HeadName[]) {
    out[name] = results[name].data as Float32Array;
  }
  return out;
}

function argmaxMasked(logits: Float32Array, valid: number[]): number {
  let best = valid[0];
  let bestVal = -Infinity;
  for (const i of valid) {
    if (logits[i] > bestVal) {
      bestVal = logits[i];
      best = i;
    }
  }
  return best;
}

interface DecisionContext {
  session: OrtSession;
  state: GameState;
  who: Who;
  deckSize: number;
  isFinalTurn: boolean;
}

async function act(ctx: DecisionContext, head: HeadName, valid: number[], drawnCard: number | null = null): Promise<number> {
  const feats = encodeState(ctx.state, ctx.who, ctx.deckSize, ctx.isFinalTurn, drawnCard);
  const outs = await runAllHeads(ctx.session, feats);
  return argmaxMasked(outs[head], valid);
}

async function decideCabo(ctx: DecisionContext): Promise<boolean> {
  return (await act(ctx, "cabo", [0, 1])) === 1;
}

async function decideDrawSource(ctx: DecisionContext): Promise<"pile" | "discard"> {
  return (await act(ctx, "draw_source", [0, 1])) === 1 ? "discard" : "pile";
}

async function decidePlaceOrDiscard(ctx: DecisionContext, card: number): Promise<"place" | "discard"> {
  return (await act(ctx, "place_or_discard", [0, 1], card)) === 1 ? "discard" : "place";
}

async function chooseDiscardPositions(ctx: DecisionContext, card: number): Promise<number[]> {
  const player = ctx.state.players[ctx.who];
  const group = findBestKnownGroup(player.hand, player.selfKnown);
  if (group !== null) {
    if ((await act(ctx, "use_group_discard", [0, 1], card)) === 1) return group;
  }
  const handLen = player.hand.length;
  const valid = Array.from({ length: handLen }, (_, i) => i);
  const pos = await act(ctx, "swap_target", valid, card);
  return [pos];
}

async function choosePeekPosition(ctx: DecisionContext): Promise<number> {
  const handLen = ctx.state.players[ctx.who].hand.length;
  return act(ctx, "peek_target", Array.from({ length: handLen }, (_, i) => i));
}

async function chooseSpyPosition(ctx: DecisionContext): Promise<number> {
  const opp = ctx.who === "human" ? ctx.state.players.agent : ctx.state.players.human;
  return act(ctx, "spy_target", Array.from({ length: opp.hand.length }, (_, i) => i));
}

async function decideSwapBlind(ctx: DecisionContext): Promise<boolean> {
  return (await act(ctx, "swap_blind_decide", [0, 1])) === 1;
}

async function chooseSwapBlindPositions(ctx: DecisionContext): Promise<[number, number]> {
  const ownLen = ctx.state.players[ctx.who].hand.length;
  const opp = ctx.who === "human" ? ctx.state.players.agent : ctx.state.players.human;
  const ownPos = await act(ctx, "swap_blind_own", Array.from({ length: ownLen }, (_, i) => i));
  const oppPos = await act(ctx, "swap_blind_opp", Array.from({ length: opp.hand.length }, (_, i) => i));
  return [ownPos, oppPos];
}

// ---------------------------------------------------------------------------
// Turn runner - mirrors cabo_rl/rules.py's take_turn, using engine.ts's
// mutation functions (deckDraw, placeDrawnCardFor, useSpyOppFor, ...).
// ---------------------------------------------------------------------------

export async function runNetAgentTurn(
  session: OrtSession,
  state: GameState,
  deckSize: number,
  isFinalTurn: boolean
): Promise<boolean> {
  const ctx: DecisionContext = { session, state, who: "agent", deckSize, isFinalTurn };

  if (!isFinalTurn && (await decideCabo(ctx))) {
    E.logMsg(state, "Agent declares CABO!");
    return true;
  }

  const discardPile = state.deck.discardPile;
  const source = discardPile.length > 0 ? await decideDrawSource(ctx) : "pile";

  let card: number;
  let fromDiscard: boolean;
  if (source === "discard") {
    card = discardPile.pop()!;
    fromDiscard = true;
    E.logMsg(state, "Agent takes the face-up card.");
  } else {
    card = E.deckDraw(state);
    fromDiscard = false;
  }

  if (fromDiscard) {
    const positions = await chooseDiscardPositions(ctx, card);
    E.placeDrawnCardFor(state, "agent", card, positions);
    return false;
  }

  const action = await decidePlaceOrDiscard(ctx, card);
  if (action === "place") {
    const positions = await chooseDiscardPositions(ctx, card);
    E.placeDrawnCardFor(state, "agent", card, positions);
    return false;
  }

  state.deck.discardPile.push(card);
  E.logMsg(state, `Agent draws from the pile and discards a ${card} face-up.`);
  let power: PowerKind = null;
  if (E.POWER_PEEK_OWN.has(card)) power = "peek";
  else if (E.POWER_SPY_OPP.has(card)) power = "spy";
  else if (E.POWER_SWAP_BLIND.has(card)) power = "swap";

  if (power === "peek") {
    const pos = await choosePeekPosition(ctx);
    E.usePeekOwnFor(state, "agent", pos);
    E.logMsg(state, `Agent peeks at its own position ${pos + 1} (value hidden from you).`);
  } else if (power === "spy") {
    const pos = await chooseSpyPosition(ctx);
    E.useSpyOppFor(state, "agent", pos);
    E.logMsg(state, `Agent spies on your position ${pos + 1} (value hidden from you).`);
  } else if (power === "swap") {
    if (await decideSwapBlind(ctx)) {
      const [ownPos, oppPos] = await chooseSwapBlindPositions(ctx);
      E.performBlindSwap(state, oppPos, ownPos); // performBlindSwap(state, humanIdx, agentIdx)
      E.logMsg(state, `Agent blind-swaps its position ${ownPos + 1} with your position ${oppPos + 1}.`);
    } else {
      E.logMsg(state, "Agent chooses not to use its swap power this time.");
    }
  }
  return false;
}
