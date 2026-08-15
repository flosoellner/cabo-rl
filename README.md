# Cabo RL

A custom 2-player card game (Cabo/Cambio variant) with two things living side by side in this repo:

- **`web/`** — a phone-installable PWA you actually play. Static, client-side only, no server. The opponent is a real neural network (self-play trained, exported to ONNX, running client-side via onnxruntime-web) — not the original tabular Q-agent. See [`web/README.md`](reference/cabo-web/README.md) for the original app notes (rule history, project layout).
- **`python/`** — the RL training pipeline (PyTorch) that produces the model `web/` ships. Currently a lighter self-play approach (see [`docs/roadmap.md`](docs/roadmap.md)); NFSP/Deep CFR/ReBeL are the planned upgrades, swappable in without touching `web/`'s code.

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
pip install torch numpy pytest onnx onnxruntime
pytest                                          # rules engine tests
python cabo_rl/enumerate.py                     # regenerate docs/complexity.md's numbers
python cabo_rl/train.py --deck real --episodes 150000 --out checkpoints/cabo_net_real.pt
python cabo_rl/export_onnx.py --checkpoint checkpoints/cabo_net_real.pt --out checkpoints/cabo_net.onnx
cp checkpoints/cabo_net.onnx ../web/public/models/cabo_net.onnx   # what the app actually loads
python cabo_rl/play_cli.py --checkpoint checkpoints/cabo_net_real.pt --deck real   # play it yourself first
```

PyTorch is MPS-accelerated on Apple Silicon (`torch.backends.mps.is_available()`), though CPU measured faster for this network's size (see `train.py`).
