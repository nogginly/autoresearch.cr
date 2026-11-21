# autoresearch.cr

A Crystal port of Karpathy's [`autoresearch`](https://github.com/karpathy/autoresearch), on [`tensomite`](https://github.com/nogginly/tensomite.cr) instead of Python and PyTorch.

In `autoresearch` an agent edits one training script, trains a small language model for a fixed five minutes, and keeps the edit only if bits per byte on a held-out set fell. Think of a chef allowed to change one recipe card, judged only on how the dish tastes after a fixed time on the stove. This port keeps the card, the stove timer and the tasting exactly as in the original; only the kitchen (Crystal and tensomite) changes. It is also a proving ground for tensomite: every place the recipe strains the library becomes an issue there.

See [SCOPE.md](SCOPE.md) for the goal and requirements, [PLAN.md](PLAN.md) for the milestones, [DESIGN.md](DESIGN.md) for how it is put together, and [DEVELOPMENT.md](DEVELOPMENT.md) for how to work on it.

## Status

Milestone M0 (foundation). There is no `train` or `prepare` yet. What builds and runs is the spike, `spikes/gpt_spike.cr`: a 4-layer, 0.9 M-parameter model with the original's structure and a MuonAdamW optimizer, trained on a synthetic token stream. It shows the pieces run on tensomite; later milestones check them against the original.

## Quick start

1. Install Crystal 1.22 or later. On Linux, also install OpenBLAS:
   ```bash
   sudo apt-get install libopenblas-dev
   ```
2. Install the shards and build the linter:
   ```bash
   make deps
   ```
   With [`ops`](https://github.com/nickthecook/ops), `ops up` does the same.
3. Run the specs and the linter:
   ```bash
   make test
   ```
4. Build the spike (a release build, about 30 s):
   ```bash
   make build
   ```
5. Train it for 200 steps:
   ```bash
   bin/gpt_spike 200
   ```
   It prints the parameter count, the loss every 20 steps (from about 5.54 down to below 0.01) and the time taken. `bin/gpt_spike 200 accum` takes the same steps with gradient accumulation over two micro-batches, and should print the same losses to the shown precision.

## Contributions

By invitation only for the time being.
