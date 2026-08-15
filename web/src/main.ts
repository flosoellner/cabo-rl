import * as app from "./app.js";
import { state, flow, ui } from "./app.js";

const root = document.getElementById("app")!;

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function cardSlot(opts: { label: string; clickable: boolean; selected: boolean; action?: string; pos?: number }): string {
  const classes = ["card"];
  if (opts.clickable) classes.push("card--clickable");
  if (opts.selected) classes.push("card--selected");
  const attrs = opts.clickable ? `data-action="${opts.action}" data-pos="${opts.pos}"` : "";
  return `<div class="${classes.join(" ")}" ${attrs}>${esc(opts.label)}</div>`;
}

function renderHands(): string {
  const human = state.players.human;
  const agent = state.players.agent;

  const ownClickablePhases = new Set(["choose_discard_targets", "choose_peek_target", "choose_swapblind_own"]);
  const agentClickablePhases = new Set(["choose_spy_target", "choose_swapblind_opp"]);
  const ownClickable = flow.current === "human" && ownClickablePhases.has(ui.phase);
  const agentClickable = flow.current === "human" && agentClickablePhases.has(ui.phase);

  const agentCards = agent.hand
    .map((_, i) =>
      cardSlot({
        label: "",
        clickable: agentClickable,
        selected: false,
        action: "agent-card",
        pos: i,
      })
    )
    .join("");

  const ownCards = human.hand
    .map((v, i) => {
      const label = human.selfKnown[i] ? String(v) : "?";
      const selected = ui.phase === "choose_discard_targets" && ui.selectedDiscardPositions.includes(i);
      return cardSlot({ label, clickable: ownClickable, selected, action: "own-card", pos: i });
    })
    .join("");

  return `
    <div class="hand-block">
      <div class="hand-label">Agent's hand</div>
      <div class="hand-row">${agentCards}</div>
    </div>
    <div class="piles">
      <div class="pile ${flow.current === "human" && ui.phase === "choose_action" ? "pile--clickable" : ""}"
           ${flow.current === "human" && ui.phase === "choose_action" ? 'data-action="draw"' : ""}>
        <div class="pile-card pile-card--back">${state.deck.drawPile.length}</div>
        <div class="pile-label">draw pile</div>
      </div>
      <div class="pile ${
        flow.current === "human" && ui.phase === "choose_action" && state.deck.discardPile.length > 0 ? "pile--clickable" : ""
      }"
           ${flow.current === "human" && ui.phase === "choose_action" && state.deck.discardPile.length > 0 ? 'data-action="take-discard"' : ""}>
        <div class="pile-card pile-card--discard">${
          state.deck.discardPile.length > 0 ? state.deck.discardPile[state.deck.discardPile.length - 1] : "-"
        }</div>
        <div class="pile-label">discard</div>
      </div>
    </div>
    <div class="hand-block">
      <div class="hand-label">Your hand</div>
      <div class="hand-row">${ownCards}</div>
    </div>
  `;
}

function renderActionPanel(): string {
  if (ui.phase === "round_over") {
    const rp = ui.roundPoints!;
    return `
      <div class="panel-text">Round result -> You: +${rp.human} &nbsp; Agent: +${rp.agent}</div>
      <div class="panel-text">Overall score -> You: ${state.players.human.totalScore} &nbsp; Agent: ${state.players.agent.totalScore}</div>
      <div class="panel-text">Next round starting&hellip;</div>
    `;
  }
  if (ui.phase === "game_over") {
    const youWon = ui.winner === "human";
    return `
      <div class="panel-text panel-text--big">${youWon ? "You win!" : "The agent wins."}</div>
      <button data-action="new-game-after-loss">new game</button>
    `;
  }

  if (flow.current === "agent") {
    return `<div class="panel-text">Agent is thinking&hellip;</div>`;
  }

  switch (ui.phase) {
    case "choose_action":
      return `
        <div class="panel-text">Click a pile to draw, or:</div>
        <button data-action="call-cabo" ${flow.isFinalTurn ? "disabled" : ""}>call cabo</button>
      `;
    case "drawn_decide":
      return `
        <div class="panel-text">You drew a ${ui.pendingCard} (only you can see this).</div>
        <button data-action="drawn-place">place it&hellip;</button>
        ${ui.pendingFromDiscard ? "" : '<button data-action="drawn-discard">discard it</button>'}
      `;
    case "choose_discard_targets":
      return `
        <div class="panel-text">Click 1-4 of your own cards to discard, then confirm. Your drawn ${ui.pendingCard} takes their place. If you pick more than one, they must all currently match each other.</div>
        <button data-action="confirm-discard-targets" ${ui.selectedDiscardPositions.length < 1 ? "disabled" : ""}>confirm</button>
        <button data-action="cancel-discard-targets">cancel</button>
      `;
    case "choose_peek_target":
      return `<div class="panel-text">Peek power: click one of your own cards to look at it.</div>`;
    case "choose_spy_target":
      return `<div class="panel-text">Spy power: click one of the agent's cards to look at it.</div>`;
    case "choose_swapblind_decide":
      return `
        <div class="panel-text">Swap power: blind-swap one of your cards with one of the agent's?</div>
        <button data-action="swapblind-yes">yes</button>
        <button data-action="swapblind-no">no</button>
      `;
    case "choose_swapblind_own":
      return `<div class="panel-text">Click one of YOUR cards to give away.</div>`;
    case "choose_swapblind_opp":
      return `<div class="panel-text">Click one of the AGENT's cards to take (blind).</div>`;
    default:
      return "";
  }
}

function renderLog(): string {
  const recent = state.log.slice(-6);
  return `<div class="log">${recent.map((m) => `<div class="log-line">${esc(m)}</div>`).join("")}</div>`;
}

function render(): void {
  root.innerHTML = `
    <div class="board">
      <div class="scorebar">
        <div class="score-badge"><div class="score-label">you</div><div class="score-value">${state.players.human.totalScore}</div></div>
        <div class="score-badge"><div class="score-label">agent</div><div class="score-value">${state.players.agent.totalScore}</div></div>
      </div>
      ${renderLog()}
      ${renderHands()}
      <div class="action-panel">${renderActionPanel()}</div>
    </div>
  `;
}

root.addEventListener("click", (e) => {
  const target = (e.target as HTMLElement).closest("[data-action]") as HTMLElement | null;
  if (!target || target.hasAttribute("disabled")) return;
  const action = target.getAttribute("data-action");
  const posAttr = target.getAttribute("data-pos");
  const pos = posAttr !== null ? parseInt(posAttr, 10) : null;

  switch (action) {
    case "draw":
      app.onDrawFromPile();
      break;
    case "take-discard":
      app.onTakeDiscard();
      break;
    case "call-cabo":
      app.onCallCabo();
      break;
    case "drawn-place":
      app.onDrawnDecidePlace();
      break;
    case "drawn-discard":
      app.onDrawnDecideDiscard();
      break;
    case "swapblind-yes":
      app.onSwapBlindDecide(true);
      break;
    case "swapblind-no":
      app.onSwapBlindDecide(false);
      break;
    case "own-card":
      if (pos !== null) app.onOwnCardClick(pos);
      break;
    case "agent-card":
      if (pos !== null) app.onAgentCardClick(pos);
      break;
    case "confirm-discard-targets":
      app.onConfirmDiscardTargets();
      break;
    case "cancel-discard-targets":
      app.onCancelDiscardTargets();
      break;
    case "new-game-after-loss":
      app.onNewGameAfterLoss();
      break;
    default:
      break;
  }
});

app.setRenderer(render);
app.setScheduler((fn) => setTimeout(fn, 550));
app.setRoundTransitionScheduler((fn) => setTimeout(fn, 2200));
app.startNewGame();
