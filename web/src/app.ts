import * as E from "./engine.js";
import { loadBrain, saveBrain } from "./storage.js";
import type { GameState, Who, PowerKind } from "./engine.js";

// ---------------------------------------------------------------------------
// UI state
// ---------------------------------------------------------------------------

type Phase =
  | "choose_action"
  | "drawn_decide"
  | "choose_discard_targets"
  | "choose_peek_target"
  | "choose_spy_target"
  | "choose_swapblind_decide"
  | "choose_swapblind_own"
  | "choose_swapblind_opp"
  | "round_over"
  | "game_over";

interface FlowState {
  current: Who;
  caboCaller: Who | null;
  isFinalTurn: boolean;
  startingPlayerThisRound: Who;
}

interface UIState {
  phase: Phase;
  pendingCard: number | null;
  pendingFromDiscard: boolean;
  selectedDiscardPositions: number[];
  swapBlindOwnPos: number | null;
  roundPoints: Record<Who, number> | null;
  winner: Who | null;
  nextStarter: Who | null;
  // Snapshot of both hands at the moment a round ends, taken before the
  // next round's deal overwrites them - so the round-over screen has
  // something real to show, not just the point delta.
  revealedHands: Record<Who, number[]> | null;
}

export let state: GameState = E.newGameState(loadBrain());
export let flow: FlowState = { current: "human", caboCaller: null, isFinalTurn: false, startingPlayerThisRound: "human" };
export let ui: UIState = {
  phase: "choose_action",
  pendingCard: null,
  pendingFromDiscard: false,
  selectedDiscardPositions: [],
  swapBlindOwnPos: null,
  roundPoints: null,
  winner: null,
  nextStarter: null,
  revealedHands: null,
};

// render() is overridden by main.ts in the browser. Left as a no-op here so
// the orchestration logic below can be exercised directly under Node for
// testing, with no DOM available.
export let render: () => void = () => {};
export function setRenderer(fn: () => void): void {
  render = fn;
}

// A tiny hook so the browser entry point can add pacing between the human's
// move and the agent's reply; tests can override this to run synchronously.
export let scheduleAgentTurn: (fn: () => void) => void = (fn) => fn();
export function setScheduler(fn: (cb: () => void) => void): void {
  scheduleAgentTurn = fn;
}

// Same idea, but for the pause between one round ending and the next one
// starting automatically (no "play again?" prompt - the match just runs
// until someone wins).
export let scheduleRoundTransition: (fn: () => void) => void = (fn) => fn();
export function setRoundTransitionScheduler(fn: (cb: () => void) => void): void {
  scheduleRoundTransition = fn;
}

// ---------------------------------------------------------------------------
// Game / round lifecycle
// ---------------------------------------------------------------------------

export function startNewGame(): void {
  state = E.newGameState(loadBrain());
  const startingPlayer: Who = Math.random() < 0.5 ? "human" : "agent";
  startRound(startingPlayer);
}

export function startRound(startingPlayer: Who): void {
  E.dealNewRound(state);
  flow = { current: startingPlayer, caboCaller: null, isFinalTurn: false, startingPlayerThisRound: startingPlayer };
  ui = {
    phase: "choose_action",
    pendingCard: null,
    pendingFromDiscard: false,
    selectedDiscardPositions: [],
    swapBlindOwnPos: null,
    roundPoints: null,
    winner: null,
    nextStarter: null,
    revealedHands: null,
  };
  render();
  if (startingPlayer === "agent") {
    scheduleAgentTurn(runAgentTurn);
  }
}

function endRound(): void {
  // Snapshot both hands before anything else touches them - the next
  // round's deal will overwrite state.players.*.hand, but the round-over
  // screen needs to keep showing what was actually revealed until the
  // player is done looking at it.
  const revealedHands: Record<Who, number[]> = {
    human: [...state.players.human.hand],
    agent: [...state.players.agent.hand],
  };
  const roundPoints = E.resolveRound(state, flow.caboCaller);
  const winner = E.applyRoundScores(state, roundPoints);
  E.agentLearn(state, roundPoints.human - roundPoints.agent);
  saveBrain(state.agentBrain);
  const nextStarter = E.determineNextStarter(roundPoints, flow.caboCaller, flow.startingPlayerThisRound);
  ui.phase = winner ? "game_over" : "round_over";
  ui.roundPoints = roundPoints;
  ui.winner = winner;
  ui.nextStarter = nextStarter;
  ui.revealedHands = revealedHands;
  render();
  // No auto-advance timer - round_over now waits for an explicit click
  // (onContinueAfterRound) so results are never yanked away before you've
  // seen them. The match still never asks "play another round?" once
  // you've clicked past this - it only gates the reveal itself.
}

export function onContinueAfterRound(): void {
  if (ui.phase !== "round_over" || ui.nextStarter === null) return;
  scheduleRoundTransition(() => startRound(ui.nextStarter!));
}

function runAgentTurn(): void {
  const calledCabo = E.agentTurn(state, flow.isFinalTurn);
  if (calledCabo) {
    flow.caboCaller = "agent";
    flow.isFinalTurn = true;
    flow.current = "human";
    ui.phase = "choose_action";
    render();
    return;
  }
  if (flow.isFinalTurn) {
    endRound();
    return;
  }
  flow.current = "human";
  ui.phase = "choose_action";
  render();
}

function finishHumanAction(): void {
  ui.phase = "choose_action";
  ui.pendingCard = null;
  ui.pendingFromDiscard = false;
  ui.swapBlindOwnPos = null;
  ui.selectedDiscardPositions = [];
  if (flow.isFinalTurn) {
    endRound();
    return;
  }
  flow.current = "agent";
  render();
  scheduleAgentTurn(runAgentTurn);
}

// ---------------------------------------------------------------------------
// Human action handlers - called by the click-delegation in main.ts (or
// directly, e.g. from tests)
// ---------------------------------------------------------------------------

export function onCallCabo(): void {
  // Rule: once either side has called Cabo, the responding player only
  // gets a single mandatory turn and may NOT call Cabo again themselves -
  // enforced here by the isFinalTurn guard (the button is also hidden in
  // the UI during that turn, this is the belt-and-suspenders check).
  if (ui.phase !== "choose_action" || flow.current !== "human" || flow.isFinalTurn) return;
  flow.caboCaller = "human";
  flow.isFinalTurn = true;
  flow.current = "agent";
  E.logMsg(state, "You call CABO! The agent gets one last regular turn, then hands are revealed.");
  render();
  scheduleAgentTurn(runAgentTurn);
}

export function onDrawFromPile(): void {
  if (ui.phase !== "choose_action" || flow.current !== "human") return;
  ui.pendingCard = E.humanDrawFromPile(state);
  ui.pendingFromDiscard = false;
  ui.phase = "drawn_decide";
  render();
}

export function onTakeDiscard(): void {
  if (ui.phase !== "choose_action" || flow.current !== "human") return;
  if (state.deck.discardPile.length === 0) return;
  ui.pendingCard = E.humanTakeDiscardCard(state);
  ui.pendingFromDiscard = true;
  ui.phase = "drawn_decide";
  render();
}

export function onDrawnDecidePlace(): void {
  if (ui.phase !== "drawn_decide" || ui.pendingCard === null) return;
  ui.selectedDiscardPositions = [];
  ui.phase = "choose_discard_targets";
  render();
}

export function onDrawnDecideDiscard(): void {
  // Only available for a card drawn face-down - a card taken from the
  // visible discard pile can't be tossed back unresolved.
  if (ui.phase !== "drawn_decide" || ui.pendingCard === null || ui.pendingFromDiscard) return;
  const power: PowerKind = E.humanDiscardDrawn(state, ui.pendingCard);
  ui.pendingCard = null;
  if (power === "peek") {
    ui.phase = "choose_peek_target";
    render();
  } else if (power === "spy") {
    ui.phase = "choose_spy_target";
    render();
  } else if (power === "swap") {
    ui.phase = "choose_swapblind_decide";
    render();
  } else {
    finishHumanAction();
  }
}

export function onSwapBlindDecide(wantsSwap: boolean): void {
  if (ui.phase !== "choose_swapblind_decide") return;
  if (!wantsSwap) {
    finishHumanAction();
    return;
  }
  ui.phase = "choose_swapblind_own";
  render();
}

export function onOwnCardClick(pos: number): void {
  if (flow.current !== "human") return;
  switch (ui.phase) {
    case "choose_discard_targets": {
      const idx = ui.selectedDiscardPositions.indexOf(pos);
      if (idx >= 0) ui.selectedDiscardPositions.splice(idx, 1);
      else ui.selectedDiscardPositions.push(pos);
      render();
      break;
    }
    case "choose_peek_target":
      E.humanUsePeekOwn(state, pos);
      finishHumanAction();
      break;
    case "choose_swapblind_own":
      ui.swapBlindOwnPos = pos;
      ui.phase = "choose_swapblind_opp";
      render();
      break;
    default:
      break;
  }
}

export function onAgentCardClick(pos: number): void {
  if (flow.current !== "human") return;
  switch (ui.phase) {
    case "choose_spy_target":
      E.humanUseSpyOpp(state, pos);
      finishHumanAction();
      break;
    case "choose_swapblind_opp":
      if (ui.swapBlindOwnPos === null) return;
      E.humanUseSwapBlind(state, ui.swapBlindOwnPos, pos);
      finishHumanAction();
      break;
    default:
      break;
  }
}

export function onConfirmDiscardTargets(): void {
  if (ui.phase !== "choose_discard_targets" || ui.selectedDiscardPositions.length < 1 || ui.pendingCard === null) return;
  // Result (success or failure) doesn't change what happens next - either
  // way the drawn card has now been fully resolved and the turn is over.
  E.humanPlaceDrawnCard(state, ui.pendingCard, [...ui.selectedDiscardPositions]);
  finishHumanAction();
}

export function onCancelDiscardTargets(): void {
  if (ui.phase !== "choose_discard_targets") return;
  ui.selectedDiscardPositions = [];
  ui.phase = "drawn_decide";
  render();
}

export function onNewGameAfterLoss(): void {
  if (ui.phase !== "game_over") return;
  startNewGame();
}
