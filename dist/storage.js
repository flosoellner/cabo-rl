// Browser persistence for the agent's learned brain. Kept separate from
// engine.ts so the engine itself has zero environment dependencies - a
// future native shell (Capacitor's Preferences plugin, React Native's
// AsyncStorage, etc.) only needs to replace this one small file.
import { defaultQTables, newAgentBrain } from "./engine.js";
const BRAIN_KEY = "cabo_agent_brain_v1";
export function loadBrain() {
    const brain = newAgentBrain();
    try {
        const raw = typeof localStorage !== "undefined" ? localStorage.getItem(BRAIN_KEY) : null;
        if (raw) {
            const data = JSON.parse(raw);
            brain.qtables = data.qtables ?? defaultQTables();
            brain.gamesPlayed = data.gamesPlayed ?? 0;
        }
    }
    catch (e) {
        // Corrupt or inaccessible storage - just start with a fresh brain.
    }
    return brain;
}
export function saveBrain(brain) {
    try {
        if (typeof localStorage === "undefined")
            return;
        localStorage.setItem(BRAIN_KEY, JSON.stringify({ qtables: brain.qtables, gamesPlayed: brain.gamesPlayed }));
    }
    catch (e) {
        console.warn("Could not save agent learning progress:", e);
    }
}
