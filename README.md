# Cabo RL

A custom 2-player card game (Cabo/Cambio variant) with two things living side by side in this repo:

- **`web/`** — a phone-installable PWA you actually play. Static, client-side only, no server. See [`web/README.md`](reference/cabo-web/README.md) for the original app notes (rule history, project layout).
- **`python/`** — a research-grade RL pipeline (PyTorch) working toward NFSP → Deep CFR → ReBeL agents, replacing the toy tabular Q-agent the app ships with today.

See [`docs/roadmap.md`](docs/roadmap.md) for what's built vs. planned, and [`docs/complexity.md`](docs/complexity.md) for measured game-complexity numbers. `reference/` holds the untouched original implementations (`cabo_cli.py`, `cabo-web/`) this repo grew out of.

## Play it

Open the deployed URL (see repo settings / GitHub Pages once enabled) on your phone, then Share → "Add to Home Screen" for a full-screen, offline-capable app icon.

To run locally:

```bash
cd web && python3 -m http.server 8766
```

then open `http://localhost:8766`.

## Develop

**Web app** (TypeScript, compiles to plain JS, no framework):

```bash
cd web
npm install
npm test   # tsc + both test suites (engine + app orchestration)
```

**Python research pipeline**:

```bash
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy pytest
pytest                              # rules engine tests
python cabo_rl/enumerate.py         # regenerate docs/complexity.md's numbers
```

PyTorch is MPS-accelerated on Apple Silicon (`torch.backends.mps.is_available()`).
