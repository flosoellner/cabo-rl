import assert from "node:assert/strict";
import * as E from "./dist/engine.js";

function freshState() {
  return E.newGameState(E.newAgentBrain());
}

// --- deck composition ---
{
  const counts = {};
  for (const v of E.CARD_VALUES) counts[v] = (counts[v] ?? 0) + 1;
  assert.equal(E.CARD_VALUES.length, 52);
  for (let v = 1; v <= 12; v++) assert.equal(counts[v], 4, `value ${v} should have 4 copies`);
  assert.equal(counts[0], 2);
  assert.equal(counts[13], 2);
  console.log("deck composition: OK");
}

// --- perform_blind_swap sync ---
{
  const state = freshState();
  state.players.human.hand = [1, 2, 3, 4];
  state.players.human.selfKnown = [true, true, false, true];
  state.players.agent.hand = [9, 8, 7, 6];
  state.players.agent.selfKnown = [true, false, true, true];
  state.agentBrain.oppKnown = [true, false, false, true];

  E.performBlindSwap(state, 0, 2);
  assert.equal(state.players.human.hand[0], 7);
  assert.equal(state.players.agent.hand[2], 1);
  assert.equal(state.players.human.selfKnown[0], false);
  assert.equal(state.players.agent.selfKnown[2], false);
  assert.equal(state.agentBrain.oppKnown[0], false);
  console.log("blind swap sync: OK");
}

// --- handPop / handAppend keep oppKnown aligned ---
{
  const state = freshState();
  state.players.human.hand = [1, 1, 5, 9];
  state.players.human.selfKnown = [true, true, true, false];
  state.agentBrain.oppKnown = [true, true, false, false];
  const val = E.handPop(state, "human", 1);
  assert.equal(val, 1);
  assert.deepEqual(state.players.human.hand, [1, 5, 9]);
  assert.deepEqual(state.players.human.selfKnown, [true, true, false]);
  assert.deepEqual(state.agentBrain.oppKnown, [true, false, false]);
  E.handAppend(state, "human", 42, true);
  assert.deepEqual(state.players.human.hand, [1, 5, 9, 42]);
  assert.deepEqual(state.agentBrain.oppKnown, [true, false, false, false]);
  console.log("handPop/handAppend sync: OK");
}

// --- kamikaze scoring ---
{
  const state = freshState();
  state.players.human.hand = [12, 12, 13, 13];
  state.players.agent.hand = [1, 2, 3, 4];
  const rp = E.resolveRound(state, "human");
  assert.deepEqual(rp, { human: 0, agent: 50 });
  console.log("kamikaze scoring: OK");
}

// --- cabo penalty scoring ---
{
  const state = freshState();
  state.players.human.hand = [1, 2, 3, 4]; // sum 10
  state.players.agent.hand = [1, 1, 1, 1]; // sum 4
  const rp = E.resolveRound(state, "human"); // human loses (10>4), human called cabo -> +5
  assert.deepEqual(rp, { human: 15, agent: 0 });
  console.log("cabo-penalty scoring: OK");
}

// --- tie scoring ---
{
  const state = freshState();
  state.players.human.hand = [5, 5];
  state.players.agent.hand = [3, 7];
  const rp = E.resolveRound(state, "agent");
  assert.deepEqual(rp, { human: 10, agent: 0 });
  console.log("tie scoring: OK");
}

// --- agentTryMatchDrawn shrinks hand whenever it holds a known duplicate,
//     regardless of the drawn card's own value ---
{
  const state = freshState();
  state.players.agent.hand = [3, 3, 7, 9];
  state.players.agent.selfKnown = [true, true, true, false];
  state.deck.drawPile = [5, 5, 5];
  state.deck.discardPile = [2];
  const did = E.agentTryMatchDrawn(state, 6); // drew a 6 - unrelated to its known 3s
  assert.equal(did, true);
  assert.equal(state.players.agent.hand.length, 3);
  assert.ok(state.players.agent.hand.includes(6), "the drawn 6 should now be in hand");
  console.log("agent match-on-draw shrink (drawn value irrelevant to the match): OK ->", state.players.agent.hand);
}

// --- agentTryMatchDrawn does nothing when there's no known duplicate at all ---
{
  const state = freshState();
  state.players.agent.hand = [3, 4, 7, 9];
  state.players.agent.selfKnown = [true, true, true, false];
  const did = E.agentTryMatchDrawn(state, 3); // drawn value matches one card, but that's not a duplicate pair
  assert.equal(did, false);
  assert.equal(state.players.agent.hand.length, 4);
  console.log("agent match-on-draw no-op with no known duplicate: OK");
}

// --- humanPlaceDrawnCard: the exact reported scenario - drew a 6, discard
//     two 7s that match each other (NOT the drawn 6) ---
{
  const state = freshState();
  state.players.human.hand = [7, 7, 2, 9];
  const result = E.humanPlaceDrawnCard(state, 6, [0, 1]);
  assert.equal(result.success, true, "two matching 7s should succeed regardless of the drawn card's value");
  assert.equal(state.players.human.hand.length, 3);
  assert.ok(state.players.human.hand.includes(6), "the drawn 6 should now be in hand");
  assert.ok(!state.players.human.hand.includes(7), "both 7s should be gone");
  console.log("humanPlaceDrawnCard (reported scenario, two 7s vs a drawn 6): OK ->", state.players.human.hand);
}

// --- humanPlaceDrawnCard: a single selected card always succeeds, no
//     equality check possible/needed - this is just the normal swap ---
{
  const state = freshState();
  state.players.human.hand = [1, 2, 3, 4];
  const result = E.humanPlaceDrawnCard(state, 9, [2]);
  assert.equal(result.success, true);
  assert.deepEqual(state.players.human.hand, [1, 2, 9, 4]);
  console.log("humanPlaceDrawnCard (single card, plain swap in place): OK ->", state.players.human.hand);
}

// --- humanPlaceDrawnCard: selecting cards that do NOT match each other
//     fails, and the drawn card is discarded as the penalty ---
{
  const state = freshState();
  state.players.human.hand = [7, 9, 2, 1];
  const before = [...state.players.human.hand];
  const result = E.humanPlaceDrawnCard(state, 6, [0, 1]); // 7 and 9 don't match each other
  assert.equal(result.success, false);
  assert.deepEqual(state.players.human.hand, before, "hand should be untouched on failure");
  assert.equal(state.deck.discardPile[state.deck.discardPile.length - 1], 6, "the drawn card should be discarded");
  console.log("humanPlaceDrawnCard (mismatched multi-select, fails cleanly): OK");
}

// --- humanPlaceDrawnCard: three-of-a-kind also works ---
{
  const state = freshState();
  state.players.human.hand = [5, 5, 5, 8];
  const result = E.humanPlaceDrawnCard(state, 1, [0, 1, 2]);
  assert.equal(result.success, true);
  assert.equal(state.players.human.hand.length, 2);
  assert.deepEqual(state.players.human.hand.sort((a, b) => a - b), [1, 8]);
  console.log("humanPlaceDrawnCard (three-of-a-kind): OK ->", state.players.human.hand);
}
{
  const state = freshState();
  state.players.human.totalScore = 95;
  let w = E.applyRoundScores(state, { human: 5, agent: 3 });
  assert.equal(state.players.human.totalScore, 50);
  assert.equal(w, null);

  state.players.agent.totalScore = 98;
  w = E.applyRoundScores(state, { human: 0, agent: 4 });
  assert.equal(state.players.agent.totalScore, 102);
  assert.equal(w, "human");
  console.log("score threshold logic: OK");
}

// --- determineNextStarter ---
{
  assert.equal(E.determineNextStarter({ human: 0, agent: 12 }, "agent", "human"), "human");
  assert.equal(E.determineNextStarter({ human: 8, agent: 0 }, "human", "human"), "agent");
  assert.equal(E.determineNextStarter({ human: 0, agent: 0 }, "human", "agent"), "human");
  assert.equal(E.determineNextStarter({ human: 0, agent: 0 }, null, "human"), "agent");
  console.log("determineNextStarter: OK");
}

// --- cabo decision no longer coin-flips on a clearly bad hand ---
{
  let calls = 0;
  const trials = 300;
  for (let i = 0; i < trials; i++) {
    const state = freshState();
    state.agentBrain.gamesPlayed = 2;
    state.players.agent.hand = [11, 12, 12, 6];
    state.players.agent.selfKnown = [true, true, false, false];
    state.players.human.hand = [4, 5, 0, 0];
    state.agentBrain.oppKnown = [false, false, false, false];
    state.deck.drawPile = new Array(43).fill(0);
    if (E.agentDecideCabo(state)) calls++;
  }
  console.log(`cabo prior sanity: called ${calls}/${trials} times with a bad hand (expect well under half)`);
  assert.ok(calls < trials * 0.35, "agent should not be coin-flipping cabo on an obviously bad hand");
}

// --- full multi-round simulation, both sides played by scripted heuristics, just checking no exceptions ---
{
  function scriptedHumanTurn(state, isFinal) {
    // very simple scripted "human": call cabo sometimes, otherwise draw
    // from either pile, occasionally attempt a (frequently failing) match
    // against the drawn card, otherwise swap/discard normally.
    if (!isFinal && Math.random() < 0.05) {
      return true; // call cabo
    }

    let card, fromDiscard;
    if (Math.random() < 0.3 && state.deck.discardPile.length > 0) {
      card = E.humanTakeDiscardCard(state);
      fromDiscard = true;
    } else {
      card = E.humanDrawFromPile(state);
      fromDiscard = false;
    }

    if (Math.random() < 0.2 && state.players.human.hand.length > 0) {
      // Occasionally try a multi-card discard, sometimes randomly guessing
      // 1-2 positions (frequently a mismatch, exercising the failure path).
      const n = 1 + Math.floor(Math.random() * 2);
      const positions = [];
      for (let k = 0; k < n && k < state.players.human.hand.length; k++) positions.push(k);
      E.humanPlaceDrawnCard(state, card, positions);
      return false;
    }

    if (Math.random() < 0.5) {
      const pos = Math.floor(Math.random() * state.players.human.hand.length);
      E.humanPlaceCardInto(state, card, pos);
    } else if (!fromDiscard) {
      const power = E.humanDiscardDrawn(state, card);
      if (power === "peek") {
        E.humanUsePeekOwn(state, Math.floor(Math.random() * state.players.human.hand.length));
      } else if (power === "spy") {
        E.humanUseSpyOpp(state, Math.floor(Math.random() * state.players.agent.hand.length));
      } else if (power === "swap") {
        if (Math.random() < 0.5) {
          const hpos = Math.floor(Math.random() * state.players.human.hand.length);
          const apos = Math.floor(Math.random() * state.players.agent.hand.length);
          E.humanUseSwapBlind(state, hpos, apos);
        }
      }
    } else {
      // took the discard card but decided against swapping it in - still
      // must resolve it somehow, so place it after all.
      const pos = Math.floor(Math.random() * state.players.human.hand.length);
      E.humanPlaceCardInto(state, card, pos);
    }
    return false;
  }

  function playRound(state, startingPlayer) {
    let current = startingPlayer;
    let caboCaller = null;
    let guard = 0;
    while (guard++ < 500) {
      let calledCabo;
      if (current === "human") calledCabo = scriptedHumanTurn(state, false);
      else calledCabo = E.agentTurn(state, false);

      if (calledCabo) {
        caboCaller = current;
        const other = current === "human" ? "agent" : "human";
        if (other === "human") scriptedHumanTurn(state, true);
        else E.agentTurn(state, true);
        break;
      }
      current = current === "human" ? "agent" : "human";
    }
    return { roundPoints: E.resolveRound(state, caboCaller), caboCaller };
  }

  const state = freshState();
  let starting = "human";
  let rounds = 0;
  let winner = null;
  for (let i = 0; i < 40 && !winner; i++) {
    E.dealNewRound(state);
    const { roundPoints, caboCaller } = playRound(state, starting);
    winner = E.applyRoundScores(state, roundPoints);
    E.agentLearn(state, roundPoints.human - roundPoints.agent);
    starting = E.determineNextStarter(roundPoints, caboCaller, starting);
    rounds++;
  }
  console.log(`full simulation: ${rounds} rounds completed, winner=${winner}, no exceptions thrown. OK`);
  assert.ok(rounds > 0);
}

console.log("\nALL ENGINE TESTS PASSED");
