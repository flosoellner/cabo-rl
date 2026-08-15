import * as E from "./engine.js";
import { loadBrain, saveBrain } from "./storage.js";
export let state = E.newGameState(loadBrain());
export let flow = { current: "human", caboCaller: null, isFinalTurn: false, startingPlayerThisRound: "human" };
export let ui = {
    phase: "choose_action",
    pendingCard: null,
    pendingFromDiscard: false,
    selectedDiscardPositions: [],
    swapBlindOwnPos: null,
    roundPoints: null,
    winner: null,
    nextStarter: null,
};
// render() is overridden by main.ts in the browser. Left as a no-op here so
// the orchestration logic below can be exercised directly under Node for
// testing, with no DOM available.
export let render = () => { };
export function setRenderer(fn) {
    render = fn;
}
// A tiny hook so the browser entry point can add pacing between the human's
// move and the agent's reply; tests can override this to run synchronously.
export let scheduleAgentTurn = (fn) => fn();
export function setScheduler(fn) {
    scheduleAgentTurn = fn;
}
// Same idea, but for the pause between one round ending and the next one
// starting automatically (no "play again?" prompt - the match just runs
// until someone wins).
export let scheduleRoundTransition = (fn) => fn();
export function setRoundTransitionScheduler(fn) {
    scheduleRoundTransition = fn;
}
// ---------------------------------------------------------------------------
// Game / round lifecycle
// ---------------------------------------------------------------------------
export function startNewGame() {
    state = E.newGameState(loadBrain());
    const startingPlayer = Math.random() < 0.5 ? "human" : "agent";
    startRound(startingPlayer);
}
export function startRound(startingPlayer) {
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
    };
    render();
    if (startingPlayer === "agent") {
        scheduleAgentTurn(runAgentTurn);
    }
}
function endRound() {
    const roundPoints = E.resolveRound(state, flow.caboCaller);
    const winner = E.applyRoundScores(state, roundPoints);
    E.agentLearn(state, roundPoints.human - roundPoints.agent);
    saveBrain(state.agentBrain);
    const nextStarter = E.determineNextStarter(roundPoints, flow.caboCaller, flow.startingPlayerThisRound);
    ui.phase = winner ? "game_over" : "round_over";
    ui.roundPoints = roundPoints;
    ui.winner = winner;
    ui.nextStarter = nextStarter;
    render();
    if (!winner) {
        // The match just keeps going - no "play another round?" prompt.
        scheduleRoundTransition(() => startRound(nextStarter));
    }
}
function runAgentTurn() {
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
function finishHumanAction() {
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
export function onCallCabo() {
    // Rule: once either side has called Cabo, the responding player only
    // gets a single mandatory turn and may NOT call Cabo again themselves -
    // enforced here by the isFinalTurn guard (the button is also hidden in
    // the UI during that turn, this is the belt-and-suspenders check).
    if (ui.phase !== "choose_action" || flow.current !== "human" || flow.isFinalTurn)
        return;
    flow.caboCaller = "human";
    flow.isFinalTurn = true;
    flow.current = "agent";
    E.logMsg(state, "You call CABO! The agent gets one last regular turn, then hands are revealed.");
    render();
    scheduleAgentTurn(runAgentTurn);
}
export function onDrawFromPile() {
    if (ui.phase !== "choose_action" || flow.current !== "human")
        return;
    ui.pendingCard = E.humanDrawFromPile(state);
    ui.pendingFromDiscard = false;
    ui.phase = "drawn_decide";
    render();
}
export function onTakeDiscard() {
    if (ui.phase !== "choose_action" || flow.current !== "human")
        return;
    if (state.deck.discardPile.length === 0)
        return;
    ui.pendingCard = E.humanTakeDiscardCard(state);
    ui.pendingFromDiscard = true;
    ui.phase = "drawn_decide";
    render();
}
export function onDrawnDecidePlace() {
    if (ui.phase !== "drawn_decide" || ui.pendingCard === null)
        return;
    ui.selectedDiscardPositions = [];
    ui.phase = "choose_discard_targets";
    render();
}
export function onDrawnDecideDiscard() {
    // Only available for a card drawn face-down - a card taken from the
    // visible discard pile can't be tossed back unresolved.
    if (ui.phase !== "drawn_decide" || ui.pendingCard === null || ui.pendingFromDiscard)
        return;
    const power = E.humanDiscardDrawn(state, ui.pendingCard);
    ui.pendingCard = null;
    if (power === "peek") {
        ui.phase = "choose_peek_target";
        render();
    }
    else if (power === "spy") {
        ui.phase = "choose_spy_target";
        render();
    }
    else if (power === "swap") {
        ui.phase = "choose_swapblind_decide";
        render();
    }
    else {
        finishHumanAction();
    }
}
export function onSwapBlindDecide(wantsSwap) {
    if (ui.phase !== "choose_swapblind_decide")
        return;
    if (!wantsSwap) {
        finishHumanAction();
        return;
    }
    ui.phase = "choose_swapblind_own";
    render();
}
export function onOwnCardClick(pos) {
    if (flow.current !== "human")
        return;
    switch (ui.phase) {
        case "choose_discard_targets": {
            const idx = ui.selectedDiscardPositions.indexOf(pos);
            if (idx >= 0)
                ui.selectedDiscardPositions.splice(idx, 1);
            else
                ui.selectedDiscardPositions.push(pos);
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
export function onAgentCardClick(pos) {
    if (flow.current !== "human")
        return;
    switch (ui.phase) {
        case "choose_spy_target":
            E.humanUseSpyOpp(state, pos);
            finishHumanAction();
            break;
        case "choose_swapblind_opp":
            if (ui.swapBlindOwnPos === null)
                return;
            E.humanUseSwapBlind(state, ui.swapBlindOwnPos, pos);
            finishHumanAction();
            break;
        default:
            break;
    }
}
export function onConfirmDiscardTargets() {
    if (ui.phase !== "choose_discard_targets" || ui.selectedDiscardPositions.length < 1 || ui.pendingCard === null)
        return;
    // Result (success or failure) doesn't change what happens next - either
    // way the drawn card has now been fully resolved and the turn is over.
    E.humanPlaceDrawnCard(state, ui.pendingCard, [...ui.selectedDiscardPositions]);
    finishHumanAction();
}
export function onCancelDiscardTargets() {
    if (ui.phase !== "choose_discard_targets")
        return;
    ui.selectedDiscardPositions = [];
    ui.phase = "drawn_decide";
    render();
}
export function onNewGameAfterLoss() {
    if (ui.phase !== "game_over")
        return;
    startNewGame();
}
