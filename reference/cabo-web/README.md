# Cabo (web version)

A clickable, browser-based version of the Cabo game and its self-improving
learning agent - a TypeScript port of the original Python/CLI version,
built as a stepping stone toward a native iOS app later.

## Just play it

No install needed to *use* it - open `index.html` directly in a browser
(double-click it, or drag it into a browser window).

## Try it on your iPhone

1. On your Mac, in this folder, run a tiny local server:
   ```
   python3 -m http.server 8000
   ```
2. Find your Mac's local IP address (System Settings -> Wi-Fi -> Details),
   e.g. `192.168.1.23`.
3. On your iPhone (same Wi-Fi network), open Safari and go to
   `http://192.168.1.23:8000`.
4. Tap the Share icon -> "Add to Home Screen". You'll get a real icon and a
   full-screen, no-browser-chrome experience - no native code required.

## How the agent's learning persists

The agent's learned Q-tables are saved to the browser's `localStorage`,
scoped to whichever browser/device you're using (this is different from the
Python version's shared `cabo_agent_brain.json` file - each browser you open
this in keeps its own separate progress). If you mostly play from one
browser on one device, this is invisible; if you switch between your Mac
and your phone often, note that they'll each build up their own separate
learning history.

## If you edit the TypeScript

The shipped `dist/*.js` files are already compiled and ready to run - you
don't need Node or TypeScript installed just to play. If you want to modify
`src/*.ts` and recompile:

```
npm install -g typescript      # one-time, if you don't have tsc
tsc                            # recompiles src/*.ts -> dist/*.js
```

## Recent rule changes

- Once either side calls Cabo, the responding player's single mandatory
  turn no longer offers the Cabo option itself.
- Placing a drawn card is now a single unified action: draw or take a
  card, then click 1-4 of your own cards to discard - the drawn card
  takes the resulting slot. A single selection is just the normal swap
  (always succeeds). Selecting 2+ only requires those selected cards to
  match **each other** - not the drawn card's value, which is unrelated.
  Guessing wrong on a 2+ selection costs you the drawn card (discarded,
  no power), since it already left the pile to get here.
- Rounds now continue automatically until someone wins the match - no
  "play another round?" prompt.

## Project layout

- `src/engine.ts` - the game rules + Q-learning agent. Pure logic, no DOM,
  no storage - this is the part that's directly reusable if you move to
  React Native or another native shell later.
- `src/storage.ts` - the *only* file that touches `localStorage`. A future
  native port only needs to replace this one file (e.g. with Capacitor's
  Preferences plugin or React Native's AsyncStorage).
- `src/app.ts` - the click-driven state machine (whose turn it is, what
  the human is currently being asked to do). DOM-free by design, so it can
  be tested directly under Node.
- `src/main.ts` - the only file that touches the DOM: renders the board and
  wires up clicks.
- `test_engine.mjs`, `test_app.mjs` - plain-Node test scripts (no install
  needed) that exercise the engine and the click-flow state machine. Run
  with `node test_engine.mjs` / `node test_app.mjs` after `tsc`.

## Path to an actual iOS app later

This exact web app can be wrapped with [Capacitor](https://capacitorjs.com/)
to produce a real Xcode project installable via TestFlight/the App Store,
with only `src/storage.ts` needing a native-storage replacement. Everything
in `src/engine.ts` carries over untouched either way.
