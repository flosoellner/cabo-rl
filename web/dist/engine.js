// Cabo game engine + learning agent.
// Faithful TypeScript port of the original Python engine. Pure logic only -
// no DOM, no localStorage, no filesystem - so it can be reused by any UI
// (this web app today, a native shell such as Capacitor or React Native
// later) without changes.
export const CARD_VALUES = (() => {
    const vals = [];
    for (let copy = 0; copy < 4; copy++) {
        for (let v = 1; v <= 12; v++)
            vals.push(v);
    }
    for (let copy = 0; copy < 2; copy++) {
        vals.push(0);
        vals.push(13);
    }
    return vals;
})();
export const POWER_PEEK_OWN = new Set([7, 8]);
export const POWER_SPY_OPP = new Set([9, 10]);
export const POWER_SWAP_BLIND = new Set([11, 12]);
export const MAX_SCORE = 100;
export const RESET_SCORE = 50;
export const CABO_PENALTY = 5;
export const KAMIKAZE_BONUS = 50;
export const ALPHA = 0.3;
export const EPSILON_BASE = 0.25;
export const EPSILON_MIN = 0.05;
// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------
export function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
}
export function choice(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
}
// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
export function newPlayer(name) {
    return { name, hand: [], selfKnown: [], totalScore: 0 };
}
export function defaultQTables() {
    return { cabo: {}, draw_source: {}, swap_or_discard: {} };
}
export function newAgentBrain() {
    return { oppKnown: [], qtables: defaultQTables(), gamesPlayed: 0, trajectory: [] };
}
export function newDeck() {
    const drawPile = [...CARD_VALUES];
    shuffle(drawPile);
    return { drawPile, discardPile: [] };
}
export function newGameState(agentBrain) {
    return {
        deck: newDeck(),
        players: { human: newPlayer("You"), agent: newPlayer("Agent") },
        agentBrain,
        log: [],
    };
}
export function logMsg(state, msg) {
    state.log.push(msg);
}
// ---------------------------------------------------------------------------
// Deck
// ---------------------------------------------------------------------------
export function deckDraw(state) {
    if (state.deck.drawPile.length === 0)
        deckReshuffle(state);
    if (state.deck.drawPile.length === 0)
        return 5; // extreme edge case
    return state.deck.drawPile.pop();
}
export function deckReshuffle(state) {
    const discard = state.deck.discardPile;
    if (discard.length <= 1)
        return;
    const top = discard[discard.length - 1];
    const rest = discard.slice(0, -1);
    shuffle(rest);
    state.deck.drawPile = rest;
    state.deck.discardPile = [top];
    logMsg(state, "The face-down pile ran out - the discard pile was reshuffled.");
}
// ---------------------------------------------------------------------------
// Hand bookkeeping (keep parallel lists in sync)
// ---------------------------------------------------------------------------
export function handPop(state, who, idx) {
    const player = state.players[who];
    const val = player.hand.splice(idx, 1)[0];
    player.selfKnown.splice(idx, 1);
    if (who === "human")
        state.agentBrain.oppKnown.splice(idx, 1);
    return val;
}
export function handAppend(state, who, val, known) {
    const player = state.players[who];
    player.hand.push(val);
    player.selfKnown.push(known);
    if (who === "human")
        state.agentBrain.oppKnown.push(false);
}
export function handSet(state, who, idx, val, known) {
    const player = state.players[who];
    player.hand[idx] = val;
    player.selfKnown[idx] = known;
    if (who === "human")
        state.agentBrain.oppKnown[idx] = false;
}
export function performBlindSwap(state, humanIdx, agentIdx) {
    const human = state.players.human;
    const agent = state.players.agent;
    const tmp = human.hand[humanIdx];
    human.hand[humanIdx] = agent.hand[agentIdx];
    agent.hand[agentIdx] = tmp;
    human.selfKnown[humanIdx] = false;
    agent.selfKnown[agentIdx] = false;
    state.agentBrain.oppKnown[humanIdx] = false;
}
// ---------------------------------------------------------------------------
// Probability / value estimation (agent's honest belief state)
// ---------------------------------------------------------------------------
export function unseenPool(state) {
    const counter = new Map();
    for (const v of CARD_VALUES)
        counter.set(v, (counter.get(v) ?? 0) + 1);
    const agent = state.players.agent;
    const human = state.players.human;
    agent.selfKnown.forEach((known, i) => {
        if (known)
            counter.set(agent.hand[i], (counter.get(agent.hand[i]) ?? 0) - 1);
    });
    state.agentBrain.oppKnown.forEach((known, i) => {
        if (known && i < human.hand.length)
            counter.set(human.hand[i], (counter.get(human.hand[i]) ?? 0) - 1);
    });
    if (state.deck.discardPile.length > 0) {
        const top = state.deck.discardPile[state.deck.discardPile.length - 1];
        counter.set(top, (counter.get(top) ?? 0) - 1);
    }
    const pool = [];
    counter.forEach((cnt, val) => {
        for (let i = 0; i < cnt; i++)
            pool.push(val);
    });
    return pool;
}
export function expectedValue(pool) {
    if (pool.length === 0)
        return 5.0;
    return pool.reduce((a, b) => a + b, 0) / pool.length;
}
export function estimateHandValue(hand, knownFlags, avgUnknown) {
    let total = 0;
    for (let i = 0; i < hand.length; i++)
        total += knownFlags[i] ? hand[i] : avgUnknown;
    return total;
}
export function bucket(value, edges) {
    for (let i = 0; i < edges.length; i++)
        if (value <= edges[i])
            return i;
    return edges.length;
}
// ---------------------------------------------------------------------------
// Tabular Q-learning
// ---------------------------------------------------------------------------
export function getEpsilon(brain) {
    return Math.max(EPSILON_MIN, EPSILON_BASE - 0.002 * brain.gamesPlayed);
}
export function qGet(table, stateKey, actions, defaults) {
    if (!(stateKey in table)) {
        const entry = {};
        for (const a of actions)
            entry[a] = defaults && a in defaults ? defaults[a] : 0.0;
        table[stateKey] = entry;
    }
    return table[stateKey];
}
export function qChoose(table, stateKey, actions, epsilon, defaults) {
    const qvals = qGet(table, stateKey, actions, defaults);
    if (Math.random() < epsilon)
        return choice(actions);
    const bestVal = Math.max(...actions.map((a) => qvals[a]));
    const bestActions = actions.filter((a) => qvals[a] === bestVal);
    return choice(bestActions);
}
export function recordDecision(state, decisionName, stateKey, action) {
    state.agentBrain.trajectory.push([decisionName, stateKey, action]);
}
export function agentLearn(state, reward) {
    const brain = state.agentBrain;
    for (const [decisionName, stateKey, action] of brain.trajectory) {
        const table = brain.qtables[decisionName];
        const qvals = qGet(table, stateKey, [action]);
        if (!(action in qvals))
            qvals[action] = 0.0;
        qvals[action] += ALPHA * (reward - qvals[action]);
    }
    brain.trajectory = [];
    brain.gamesPlayed += 1;
}
// ---------------------------------------------------------------------------
// Agent strategic decisions (Q-learned, with heuristic priors for new states)
// ---------------------------------------------------------------------------
export function agentDecideCabo(state) {
    const brain = state.agentBrain;
    const avg = expectedValue(unseenPool(state));
    const agent = state.players.agent;
    const human = state.players.human;
    const ownEst = estimateHandValue(agent.hand, agent.selfKnown, avg);
    const oppEst = estimateHandValue(human.hand, brain.oppKnown, avg);
    const unknownOwn = agent.selfKnown.filter((k) => !k).length;
    const deckLeftBucket = bucket(state.deck.drawPile.length, [10, 25]);
    const stateKey = JSON.stringify([
        bucket(ownEst, [8, 16, 26]),
        bucket(oppEst, [8, 16, 26]),
        unknownOwn,
        deckLeftBucket,
    ]);
    const actions = ["call", "wait"];
    const margin = oppEst - ownEst - unknownOwn * 1.5;
    const defaults = { call: margin * 0.5, wait: 0.0 };
    const action = qChoose(brain.qtables.cabo, stateKey, actions, getEpsilon(brain), defaults);
    recordDecision(state, "cabo", stateKey, action);
    return action === "call";
}
export function agentDecideDrawSource(state) {
    const brain = state.agentBrain;
    const discardPile = state.deck.discardPile;
    if (discardPile.length === 0)
        return "pile";
    const top = discardPile[discardPile.length - 1];
    const avg = expectedValue(unseenPool(state));
    const agent = state.players.agent;
    const ownEst = estimateHandValue(agent.hand, agent.selfKnown, avg);
    const discardBucket = top <= 3 ? 0 : top <= 7 ? 1 : 2;
    const unknownOwn = agent.selfKnown.filter((k) => !k).length;
    const stateKey = JSON.stringify([bucket(ownEst, [8, 16, 26]), discardBucket, unknownOwn]);
    const actions = ["pile", "discard"];
    const defaults = { pile: 0.0, discard: (avg - top) * 0.6 };
    const action = qChoose(brain.qtables.draw_source, stateKey, actions, getEpsilon(brain), defaults);
    recordDecision(state, "draw_source", stateKey, action);
    return action;
}
export function agentDecideSwapOrDiscard(state, card) {
    const brain = state.agentBrain;
    const avg = expectedValue(unseenPool(state));
    const agent = state.players.agent;
    const ownEst = estimateHandValue(agent.hand, agent.selfKnown, avg);
    const cardBucket = card <= 3 ? 0 : card <= 7 ? 1 : 2;
    const hasPower = POWER_PEEK_OWN.has(card) || POWER_SPY_OPP.has(card) || POWER_SWAP_BLIND.has(card) ? 1 : 0;
    const unknownOwn = agent.selfKnown.filter((k) => !k).length;
    const stateKey = JSON.stringify([bucket(ownEst, [8, 16, 26]), cardBucket, hasPower, unknownOwn]);
    const actions = ["swap", "discard"];
    const ownAvgPerCard = ownEst / Math.max(agent.hand.length, 1);
    const defaults = { swap: (ownAvgPerCard - card) * 0.6, discard: hasPower ? 0.4 : -0.4 };
    const action = qChoose(brain.qtables.swap_or_discard, stateKey, actions, getEpsilon(brain), defaults);
    recordDecision(state, "swap_or_discard", stateKey, action);
    return action;
}
// ---------------------------------------------------------------------------
// Agent tactical / mechanical heuristics (perfect-memory driven, not learned)
// ---------------------------------------------------------------------------
export function agentChooseSwapPosition(state, card) {
    const agent = state.players.agent;
    const avg = expectedValue(unseenPool(state));
    let bestPos = 0;
    let bestGain = -Infinity;
    agent.hand.forEach((v, i) => {
        const ref = agent.selfKnown[i] ? v : avg;
        const gain = ref - card;
        if (gain > bestGain) {
            bestGain = gain;
            bestPos = i;
        }
    });
    return bestPos;
}
export function agentSwapIn(state, pos, card) {
    const agent = state.players.agent;
    const oldCard = agent.hand[pos];
    handSet(state, "agent", pos, card, true);
    state.deck.discardPile.push(oldCard);
    logMsg(state, `Agent places it at its position ${pos + 1}, discarding its old card (${oldCard}) face-up.`);
}
export function agentUsePeekOwn(state) {
    const agent = state.players.agent;
    const unknownPositions = agent.selfKnown.map((k, i) => (!k ? i : -1)).filter((i) => i >= 0);
    if (unknownPositions.length === 0) {
        logMsg(state, "Agent's peek power fizzles (it already knows all its own cards).");
        return;
    }
    const pos = choice(unknownPositions);
    agent.selfKnown[pos] = true;
    logMsg(state, `Agent peeks at its own position ${pos + 1} (value hidden from you).`);
}
export function agentUseSpyOpp(state) {
    const human = state.players.human;
    const brain = state.agentBrain;
    const unknownPositions = [];
    for (let i = 0; i < human.hand.length; i++)
        if (!brain.oppKnown[i])
            unknownPositions.push(i);
    if (unknownPositions.length === 0) {
        logMsg(state, "Agent's spy power fizzles (it already knows all of your cards).");
        return;
    }
    const pos = choice(unknownPositions);
    brain.oppKnown[pos] = true;
    logMsg(state, `Agent spies on your position ${pos + 1} (value hidden from you).`);
}
export function agentUseSwapBlind(state) {
    const agent = state.players.agent;
    const human = state.players.human;
    const brain = state.agentBrain;
    const avg = expectedValue(unseenPool(state));
    let worstAgentPos = 0;
    let worstVal = -Infinity;
    agent.hand.forEach((v, i) => {
        const ref = agent.selfKnown[i] ? v : avg;
        if (ref > worstVal) {
            worstVal = ref;
            worstAgentPos = i;
        }
    });
    const ourVal = agent.selfKnown[worstAgentPos] ? agent.hand[worstAgentPos] : avg;
    const knownHumanPositions = [];
    for (let i = 0; i < human.hand.length; i++)
        if (brain.oppKnown[i])
            knownHumanPositions.push(i);
    if (knownHumanPositions.length > 0) {
        let bestHumanPos = knownHumanPositions[0];
        for (const i of knownHumanPositions)
            if (human.hand[i] < human.hand[bestHumanPos])
                bestHumanPos = i;
        if (ourVal > human.hand[bestHumanPos]) {
            performBlindSwap(state, bestHumanPos, worstAgentPos);
            logMsg(state, `Agent blind-swaps its position ${worstAgentPos + 1} with your position ${bestHumanPos + 1}.`);
            return;
        }
    }
    if (agent.selfKnown[worstAgentPos] && agent.hand[worstAgentPos] >= 8 && human.hand.length > 0) {
        const targetPos = Math.floor(Math.random() * human.hand.length);
        performBlindSwap(state, targetPos, worstAgentPos);
        logMsg(state, `Agent blind-swaps its position ${worstAgentPos + 1} with your position ${targetPos + 1}.`);
        return;
    }
    logMsg(state, "Agent chooses not to use its swap power this time.");
}
export function agentTryMatchDrawn(state, card) {
    // Whenever the agent already knows it's holding 2+ cards of the same
    // value, it can discard them together and place the just-drawn/taken
    // card into hand as the single replacement - always a good trade since
    // it reduces hand size for free. This has nothing to do with the drawn
    // card's own value - only the discarded group needs to match itself.
    const agent = state.players.agent;
    const valuePositions = new Map();
    agent.hand.forEach((v, i) => {
        if (agent.selfKnown[i]) {
            const arr = valuePositions.get(v) ?? [];
            arr.push(i);
            valuePositions.set(v, arr);
        }
    });
    for (const [v, positions] of valuePositions) {
        if (positions.length >= 2) {
            const sorted = [...positions].sort((a, b) => b - a);
            for (const idx of sorted) {
                const discardedVal = handPop(state, "agent", idx);
                state.deck.discardPile.push(discardedVal);
            }
            handAppend(state, "agent", card, true);
            logMsg(state, `Agent discards its matching ${v}s (${positions.length} cards) and places its drawn card there instead.`);
            return true;
        }
    }
    return false;
}
// ---------------------------------------------------------------------------
// Agent turn (fully synchronous - the agent never needs to "wait")
// ---------------------------------------------------------------------------
export function agentTurn(state, final) {
    if (!final) {
        if (agentDecideCabo(state)) {
            logMsg(state, "Agent declares CABO!");
            return true;
        }
    }
    const discardPile = state.deck.discardPile;
    const topDiscard = discardPile.length > 0 ? discardPile[discardPile.length - 1] : null;
    const source = agentDecideDrawSource(state);
    let card;
    let fromDiscard;
    if (source === "discard" && topDiscard !== null) {
        card = discardPile.pop();
        fromDiscard = true;
        logMsg(state, "Agent takes the face-up card.");
    }
    else {
        card = deckDraw(state);
        fromDiscard = false;
    }
    if (agentTryMatchDrawn(state, card)) {
        return false;
    }
    if (fromDiscard) {
        const pos = agentChooseSwapPosition(state, card);
        agentSwapIn(state, pos, card);
    }
    else {
        const action = agentDecideSwapOrDiscard(state, card);
        if (action === "swap") {
            const pos = agentChooseSwapPosition(state, card);
            logMsg(state, "Agent draws from the pile.");
            agentSwapIn(state, pos, card);
        }
        else {
            state.deck.discardPile.push(card);
            logMsg(state, `Agent draws from the pile and discards a ${card} face-up.`);
            if (POWER_PEEK_OWN.has(card))
                agentUsePeekOwn(state);
            else if (POWER_SPY_OPP.has(card))
                agentUseSpyOpp(state);
            else if (POWER_SWAP_BLIND.has(card))
                agentUseSwapBlind(state);
        }
    }
    return false;
}
export function humanDrawFromPile(state) {
    return deckDraw(state);
}
export function humanTakeDiscardCard(state) {
    return state.deck.discardPile.pop();
}
export function humanPlaceCardInto(state, card, pos) {
    const human = state.players.human;
    const oldCard = human.hand[pos];
    handSet(state, "human", pos, card, true);
    state.deck.discardPile.push(oldCard);
    logMsg(state, `You place the ${card} at position ${pos + 1}; your old card (${oldCard}) is now face-up on the discard pile.`);
}
export function humanDiscardDrawn(state, card) {
    state.deck.discardPile.push(card);
    logMsg(state, `You discard the ${card} face-up.`);
    if (POWER_PEEK_OWN.has(card))
        return "peek";
    if (POWER_SPY_OPP.has(card))
        return "spy";
    if (POWER_SWAP_BLIND.has(card))
        return "swap";
    return null;
}
export function humanUsePeekOwn(state, pos) {
    const human = state.players.human;
    const val = human.hand[pos];
    human.selfKnown[pos] = true;
    logMsg(state, `Position ${pos + 1} is a ${val}.`);
    return val;
}
export function humanUseSpyOpp(state, pos) {
    const val = state.players.agent.hand[pos];
    logMsg(state, `The agent's position ${pos + 1} is a ${val}. Remember it - the game won't remind you!`);
    return val;
}
export function humanUseSwapBlind(state, hpos, apos) {
    performBlindSwap(state, hpos, apos);
    logMsg(state, `You blind-swap your position ${hpos + 1} with the agent's position ${apos + 1}. Neither value was revealed.`);
}
// This is really just "swap the drawn card into your hand" generalized:
// normally you discard exactly one old card to make room, but if you
// discard 2-4 at once they must all currently match each other (their
// value has nothing to do with the drawn card - the drawn card is simply
// what fills the resulting slot). A single selected card never needs an
// equality check since there's nothing to compare it against.
export function humanPlaceDrawnCard(state, card, positions) {
    const player = state.players.human;
    const values = positions.map((p) => player.hand[p]);
    if (positions.length === 1) {
        humanPlaceCardInto(state, card, positions[0]);
        return { success: true, positions, values };
    }
    const allEqual = values.length > 0 && values.every((v) => v === values[0]);
    if (allEqual && positions.length >= 2) {
        const sorted = [...positions].sort((a, b) => b - a);
        for (const idx of sorted) {
            const discardedVal = handPop(state, "human", idx);
            state.deck.discardPile.push(discardedVal);
        }
        handAppend(state, "human", card, true);
        logMsg(state, `Success! You discarded ${positions.length} matching ${values[0]}s and placed your drawn ${card} instead.`);
        return { success: true, positions, values };
    }
    // Failed attempt: the drawn card is already committed (it left the pile
    // to get here), so the penalty is that it's discarded outright with no
    // power and no swap; your hand is left untouched.
    state.deck.discardPile.push(card);
    logMsg(state, `Failed swap! Those cards were actually ${values.join(", ")} - not all equal to each other. Your drawn card is discarded.`);
    return { success: false, positions, values };
}
// ---------------------------------------------------------------------------
// Generalized (who-agnostic) versions of the above, added for the net-based
// agent (src/netAgent.ts) - the human_* functions above are hardcoded to
// state.players.human and stay untouched (still exercised by test_engine.mjs
// and the click-driven UI). These mirror cabo_rl/rules.py's collapsing of
// human_*/agent_* into single generic functions.
// ---------------------------------------------------------------------------
export function placeCardIntoFor(state, who, card, pos) {
    const player = state.players[who];
    const oldCard = player.hand[pos];
    handSet(state, who, pos, card, true);
    state.deck.discardPile.push(oldCard);
    logMsg(state, `${who} places it at position ${pos + 1}, discarding its old card (${oldCard}) face-up.`);
}
export function placeDrawnCardFor(state, who, card, positions) {
    const player = state.players[who];
    const values = positions.map((p) => player.hand[p]);
    if (positions.length === 1) {
        placeCardIntoFor(state, who, card, positions[0]);
        return { success: true, positions, values };
    }
    const allEqual = values.length > 0 && values.every((v) => v === values[0]);
    if (allEqual && positions.length >= 2) {
        const sorted = [...positions].sort((a, b) => b - a);
        for (const idx of sorted) {
            const discardedVal = handPop(state, who, idx);
            state.deck.discardPile.push(discardedVal);
        }
        handAppend(state, who, card, true);
        logMsg(state, `Success! ${who} discarded ${positions.length} matching ${values[0]}s and placed the drawn ${card} instead.`);
        return { success: true, positions, values };
    }
    state.deck.discardPile.push(card);
    logMsg(state, `Failed swap! Those cards were actually ${values.join(", ")} - not all equal. The drawn card is discarded.`);
    return { success: false, positions, values };
}
// No value in the log message here (unlike humanUsePeekOwn) - the only
// caller is the net agent peeking at its OWN card, and the whole point is
// that value stays hidden from the human. The caller logs its own
// "(value hidden from you)" message.
export function usePeekOwnFor(state, who, pos) {
    const player = state.players[who];
    const val = player.hand[pos];
    player.selfKnown[pos] = true;
    return val;
}
export function useSpyOppFor(state, who, pos) {
    const opp = who === "human" ? state.players.agent : state.players.human;
    const val = opp.hand[pos];
    if (who === "agent")
        state.agentBrain.oppKnown[pos] = true;
    return val;
}
// ---------------------------------------------------------------------------
// Round flow / scoring
// ---------------------------------------------------------------------------
export function dealNewRound(state) {
    state.deck = newDeck();
    for (const who of ["human", "agent"]) {
        const p = state.players[who];
        p.hand = [];
        for (let i = 0; i < 4; i++)
            p.hand.push(state.deck.drawPile.pop());
        p.selfKnown = [true, true, false, false];
    }
    state.agentBrain.oppKnown = [false, false, false, false];
    const top = state.deck.drawPile.pop();
    state.deck.discardPile.push(top);
    state.log = [];
    const h = state.players.human;
    logMsg(state, `New round. Your first two cards: position 1 = ${h.hand[0]}, position 2 = ${h.hand[1]}.`);
    logMsg(state, `The starting face-up card on the discard pile is a ${top}.`);
}
export function isKamikaze(hand) {
    const sorted = [...hand].sort((a, b) => a - b);
    return sorted.length === 4 && sorted[0] === 12 && sorted[1] === 12 && sorted[2] === 13 && sorted[3] === 13;
}
export function resolveRound(state, caboCaller) {
    const human = state.players.human;
    const agent = state.players.agent;
    const humanSum = human.hand.reduce((a, b) => a + b, 0);
    const agentSum = agent.hand.reduce((a, b) => a + b, 0);
    logMsg(state, `Your final hand: [${human.hand.join(", ")}] (sum = ${humanSum})`);
    logMsg(state, `Agent final hand: [${agent.hand.join(", ")}] (sum = ${agentSum})`);
    const humanKamikaze = isKamikaze(human.hand);
    const agentKamikaze = isKamikaze(agent.hand);
    const roundPoints = { human: 0, agent: 0 };
    if (humanKamikaze || agentKamikaze) {
        if (humanKamikaze) {
            logMsg(state, "KAMIKAZE! Your hand was exactly 12,12,13,13 - the agent takes +50!");
            roundPoints.human = 0;
            roundPoints.agent += KAMIKAZE_BONUS;
        }
        if (agentKamikaze) {
            logMsg(state, "KAMIKAZE! The agent's hand was exactly 12,12,13,13 - you take +50!");
            roundPoints.agent = 0;
            roundPoints.human += KAMIKAZE_BONUS;
        }
    }
    else if (humanSum < agentSum) {
        roundPoints.human = 0;
        roundPoints.agent = agentSum;
        if (caboCaller === "agent")
            roundPoints.agent += CABO_PENALTY;
    }
    else if (agentSum < humanSum) {
        roundPoints.agent = 0;
        roundPoints.human = humanSum;
        if (caboCaller === "human")
            roundPoints.human += CABO_PENALTY;
    }
    else {
        if (caboCaller === "human") {
            roundPoints.human = 0;
            roundPoints.agent = agentSum;
        }
        else {
            roundPoints.agent = 0;
            roundPoints.human = humanSum;
        }
    }
    return roundPoints;
}
export function applyRoundScores(state, roundPoints) {
    for (const who of ["human", "agent"])
        state.players[who].totalScore += roundPoints[who];
    for (const who of ["human", "agent"]) {
        if (state.players[who].totalScore === MAX_SCORE) {
            state.players[who].totalScore = RESET_SCORE;
            logMsg(state, `${who === "human" ? "You hit" : "The agent hit"} exactly 100 - reduced to 50!`);
        }
    }
    for (const who of ["human", "agent"]) {
        if (state.players[who].totalScore > MAX_SCORE) {
            return who === "human" ? "agent" : "human";
        }
    }
    return null;
}
export function determineNextStarter(roundPoints, caboCaller, previousStarter) {
    const zeroPlayers = ["human", "agent"].filter((w) => roundPoints[w] === 0);
    if (zeroPlayers.length === 1)
        return zeroPlayers[0];
    if (zeroPlayers.length === 2 && caboCaller)
        return caboCaller;
    return previousStarter === "human" ? "agent" : "human";
}
