import assert from "node:assert/strict";
import * as app from "./dist/app.js";

let renderCount = 0;
app.setRenderer(() => {
  renderCount++;
});
app.setScheduler((fn) => fn());
app.setRoundTransitionScheduler((fn) => fn()); // auto-continue immediately in tests too

// -------------------------------------------------------------------------
// 1) Cabo lockout: once a cabo call is in flight, the responding player's
//    single mandatory turn must not offer (or accept) another cabo call.
//    (Round-transition scheduler is deliberately deferred/captured here so
//    we can inspect state right after the call, before the next round -
//    which the real auto-continue feature would otherwise kick off
//    immediately - overwrites flow with a fresh object.)
// -------------------------------------------------------------------------
{
  let deferredTransition = null;
  app.setRoundTransitionScheduler((fn) => {
    deferredTransition = fn;
  });

  app.startNewGame();
  if (deferredTransition) {
    deferredTransition();
    deferredTransition = null;
  }
  app.flow.current = "human";
  app.flow.isFinalTurn = false;
  app.flow.caboCaller = null;
  app.ui.phase = "choose_action";

  app.onCallCabo();
  assert.equal(app.flow.caboCaller, "human");
  assert.equal(app.flow.isFinalTurn, true);
  assert.ok(["round_over", "game_over"].includes(app.ui.phase), `expected round to resolve, got ${app.ui.phase}`);
  console.log("cabo lockout (human calls, round resolves without agent re-calling): OK");

  app.setRoundTransitionScheduler((fn) => fn()); // restore synchronous default for later tests
}

{
  // Now force the OTHER direction: it's the human's mandatory final turn
  // after the agent called cabo - onCallCabo must be a no-op.
  app.startNewGame();
  app.flow.current = "human";
  app.flow.isFinalTurn = true;
  app.flow.caboCaller = "agent";
  app.ui.phase = "choose_action";

  const caboCallerBefore = app.flow.caboCaller;
  app.onCallCabo();
  assert.equal(app.flow.caboCaller, caboCallerBefore, "onCallCabo must be a no-op during the mandatory final turn");
  console.log("cabo lockout (human cannot re-call on final turn): OK");
}

// -------------------------------------------------------------------------
// 2) Unified "place drawn card" flow: draw first, then pick 1-4 of your own
//    cards to discard - the drawn card's own value is irrelevant, only the
//    selected cards need to match EACH OTHER (the exact bug the person
//    reported: drawing a 6 and discarding two 7s must succeed).
//    Agent scheduler deferred/captured throughout so its turn can't run
//    (and potentially mutate the human's hand via blind-swap) before we've
//    finished inspecting the human-side state.
// -------------------------------------------------------------------------
function withDeferredAgent(fn) {
  let deferred = null;
  app.setScheduler((cb) => {
    deferred = cb;
  });
  try {
    fn();
  } finally {
    app.setScheduler((cb) => cb()); // restore synchronous default
  }
}

withDeferredAgent(() => {
  // The exact reported scenario: drew a 6, want to discard two 7s.
  app.startNewGame();
  app.state.players.human.hand = [7, 7, 2, 1];
  app.state.players.human.selfKnown = [true, true, true, true];
  app.state.deck.drawPile = [9, 9, 6]; // .pop() takes from the end -> next draw is 6
  app.flow.current = "human";
  app.flow.isFinalTurn = false;
  app.ui.phase = "choose_action";

  app.onDrawFromPile();
  assert.equal(app.ui.phase, "drawn_decide");
  assert.equal(app.ui.pendingCard, 6);
  assert.equal(app.ui.pendingFromDiscard, false);

  app.onDrawnDecidePlace();
  assert.equal(app.ui.phase, "choose_discard_targets");

  app.onOwnCardClick(0); // first 7
  app.onOwnCardClick(1); // second 7 - these match each other, not the drawn 6
  assert.deepEqual(app.ui.selectedDiscardPositions, [0, 1]);

  app.onConfirmDiscardTargets();
  assert.equal(app.state.players.human.hand.length, 3);
  assert.ok(app.state.players.human.hand.includes(6), "the drawn 6 should now be in hand");
  assert.ok(!app.state.players.human.hand.includes(7), "both 7s should be gone");
  console.log("reported bug scenario (drew 6, discarded two 7s): OK ->", app.state.players.human.hand);
});

withDeferredAgent(() => {
  // A single selected card is just the plain swap - always succeeds.
  app.startNewGame();
  app.state.players.human.hand = [1, 2, 3, 4];
  app.state.players.human.selfKnown = [true, true, true, true];
  app.state.deck.drawPile = [5, 5, 9];
  app.flow.current = "human";
  app.flow.isFinalTurn = false;
  app.ui.phase = "choose_action";

  app.onDrawFromPile();
  app.onDrawnDecidePlace();
  app.onOwnCardClick(2);
  app.onConfirmDiscardTargets();
  assert.deepEqual(app.state.players.human.hand, [1, 2, 9, 4]);
  console.log("single-card place (plain swap): OK ->", app.state.players.human.hand);
});

withDeferredAgent(() => {
  // Taking from the discard pile also offers place/discard (but not
  // "discard it outright", since that card is already public).
  app.startNewGame();
  app.state.deck.discardPile = [7];
  app.state.players.human.hand = [1, 2, 3, 4];
  app.state.players.human.selfKnown = [true, true, true, true];
  app.flow.current = "human";
  app.flow.isFinalTurn = false;
  app.ui.phase = "choose_action";

  app.onTakeDiscard();
  assert.equal(app.ui.phase, "drawn_decide");
  assert.equal(app.ui.pendingFromDiscard, true);
  console.log("take-discard now routes through drawn_decide: OK");
});

withDeferredAgent(() => {
  // Selecting cards that don't match each other fails and the drawn card
  // is discarded as the penalty - the human's hand is untouched.
  app.startNewGame();
  app.state.players.human.hand = [4, 9, 2, 1];
  app.state.players.human.selfKnown = [true, true, true, true];
  app.state.deck.drawPile = [6, 6, 8];
  app.flow.current = "human";
  app.flow.isFinalTurn = false;
  app.ui.phase = "choose_action";

  app.onDrawFromPile();
  app.onDrawnDecidePlace();
  app.onOwnCardClick(0); // holds a 4
  app.onOwnCardClick(1); // holds a 9 - doesn't match position 0's 4
  const handBefore = [...app.state.players.human.hand];
  app.onConfirmDiscardTargets();
  assert.deepEqual(app.state.players.human.hand, handBefore, "hand should be unchanged on a failed attempt");
  assert.equal(app.state.deck.discardPile[app.state.deck.discardPile.length - 1], 8, "the drawn card should end up discarded");
  console.log("mismatched multi-select fails cleanly, drawn card discarded: OK");
});

// -------------------------------------------------------------------------
// 3) Rounds auto-continue with no "play again?" prompt until someone wins.
// -------------------------------------------------------------------------
{
  app.startNewGame();
  let guard = 0;
  let roundsSeen = 0;
  let sawGameOver = false;
  let lastPhase = null;

  while (guard++ < 4000) {
    if (app.ui.phase === "game_over") {
      sawGameOver = true;
      break;
    }
    if (app.ui.phase !== lastPhase && app.ui.phase === "choose_action" && app.flow.startingPlayerThisRound) {
      // heuristically counts a "new round" boundary; not load-bearing, just informational
    }
    lastPhase = app.ui.phase;
    if (app.flow.current !== "human") break; // shouldn't happen with sync scheduler
    switch (app.ui.phase) {
      case "choose_action": {
        const r = Math.random();
        if (r < 0.05) app.onCallCabo();
        else if (r < 0.25 && app.state.deck.discardPile.length > 0) app.onTakeDiscard();
        else app.onDrawFromPile();
        break;
      }
      case "drawn_decide":
        app.onDrawnDecidePlace();
        break;
      case "choose_discard_targets":
        app.onOwnCardClick(Math.floor(Math.random() * app.state.players.human.hand.length));
        app.onConfirmDiscardTargets();
        break;
      default:
        break;
    }
  }

  assert.ok(sawGameOver, "expected the match to reach game_over via auto-continuing rounds");
  // Crucially: at no point should the state machine have entered a phase
  // that requires a manual "continue" click - round_over is transient
  // (scheduleRoundTransition fires synchronously in this test) so it
  // should never be the LAST phase before game_over is reached this way.
  console.log(`auto-continue rounds: reached game_over after ${guard} steps, no manual continue click used: OK`);
}

console.log("\nALL APP ORCHESTRATION TESTS PASSED");
