# autoresearch.cr

A Crystal port of Karpathy's [`autoresearch`](https://github.com/karpathy/autoresearch), on [`tensomite`](https://github.com/nogginly/tensomite.cr) instead of Python and PyTorch.

In `autoresearch` an agent edits one training script, trains a small language model for a fixed five minutes, and keeps the edit only if bits per byte on a held-out set fell. Think of a chef allowed to change one recipe card, judged only on how the dish tastes after a fixed time on the stove. This port keeps the card, the stove timer and the tasting exactly as in the original; only the kitchen (Crystal and tensomite) changes. It is also a proving ground for tensomite: every place the recipe strains the library becomes an issue there.

See [SCOPE.md](SCOPE.md) for the goal and requirements, [PLAN.md](PLAN.md) for the milestones, [DESIGN.md](DESIGN.md) for how it is put together, and [DEVELOPMENT.md](DEVELOPMENT.md) for how to work on it.

## Status

Milestones M0 to M3 are done, and M4 (the training loop and metric) is in progress. What works today:

- **`bin/prepare`** downloads the dataset, trains the tokenizer and writes token shards. Its tokenizer, token ids and batches match `prepare.py`'s.
- **`train.cr`** holds the model and the MuonAdamW optimizer. Both match `train.py` within measured tolerances, on BLAS, CUDA and Metal. `bin/train` trains for the five-minute budget and evaluates `val_bpb` as `prepare.py` does.

## Quick start

1. Install Crystal 1.22 or later. On Linux, also install OpenBLAS and libzstd:
   ```bash
   sudo apt-get install libopenblas-dev libzstd-dev
   ```
   On macOS:
   ```bash
   brew install zstd pkgconf
   ```
2. Install the shards and build the linter:
   ```bash
   make deps
   ```
3. Run the specs and the linter:
   ```bash
   make test
   ```
4. Build the programs (release builds, about a minute):
   ```bash
   make build
   ```
5. Prepare the data. This downloads about 1 GB, then trains and tokenizes for about 5 minutes on four cores:
   ```bash
   bin/prepare
   ```
   It fetches 10 training shards and the validation shard, trains the tokenizer and writes token shards, all under `~/.cache/autoresearch.cr` (or `AUTORESEARCH_CACHE`). Run it again and it skips what is done.
6. Train for five minutes on the `small` profile, then evaluate:
   ```bash
   bin/train --device blas
   ```
   It prints a line per step, then the summary block `program.md` reads (`val_bpb`, the training seconds, peak memory and so on). On a GPU, use `--device cuda` (after `make build CUDA=1`) or `--device metal`. BLAS trains about 50 steps in the budget on four cores; it is for checking that things work, not for experiments.

## Contributions

By invitation only for the time being.
