# `autoresearch.cr`: scope

Nov 21, 2026

## Goal

Run Karpathy's [`autoresearch`](https://github.com/karpathy/autoresearch) experiment, an agent that edits a training script, trains for a fixed time budget and keeps or discards the change on one metric, in Crystal, on [`tensomite`](https://github.com/nogginly/tensomite.cr) instead of Python and PyTorch.

Two things follow from the goal, and the project is judged on both:

1. **A working Crystal `autoresearch`.** An agent can follow `program.md` in this repo and run keep-or-discard experiments unattended.
2. **A proving ground for `tensomite`.** A real, published training recipe (value embeddings, windowed attention, Muon, a time-budgeted loop) is run on the library. Every place it strains becomes an issue in tensomite, not a workaround here.

Speed is not a goal. The same outcome is: the same model, optimizer, schedule and metric, run for the same kind of fixed budget.

## What stays the same as the original

These are the parts that make a result mean something, so they are kept exact:

- **The metric.** `val_bpb`: bits per byte on a fixed validation set, summed over per-token losses weighted by each token's byte length, with special tokens excluded. It is vocabulary-independent, so changes of tokenizer or model size compare fairly.
- **The contract with the agent.** One editable file (`train.cr`, the counterpart of `train.py`), one fixed harness (data, tokenizer, evaluation, constants), a wall-clock training budget excluding startup, the same summary block on stdout, the same `results.tsv` and keep-or-reset loop.
- **The recipe.** The model (value embeddings, sliding-window attention, softcapped logits, per-layer scalars), the MuonAdamW optimizer with its per-group settings, and the schedules, as in `train.py` at the commit this port is pinned to.
- **The data.** The same dataset and validation shard, a tokenizer trained with the same algorithm and split pattern, and best-fit document packing.

## What may differ

- **Precision.** `F32` until tensomite computes in BF16. Differences from the original's autocast are expected and recorded, not hidden.
- **Speed and scale.** Tensomite computes on CUDA cores, without tensor cores or fused attention, and caps one tensor at 8 GiB. The reference size (depth 8, sequence 2,048) may not fit or may be slow; a scaled-down profile is a supported configuration, not a failure.
- **Absolute `val_bpb`.** The port's numbers are comparable between experiments on one machine, not with Karpathy's published 0.998.
- **Tooling.** Data preparation is a Crystal tool in this repo. Python is used only for generating reference fixtures, never to run an experiment.

## Requirements

### Functional

1. `prepare` downloads the dataset shards, trains the tokenizer and writes token shards, in one command, resumably.
2. `train` builds the model from a depth, trains for the time budget on a CPU, CUDA or Metal device, evaluates, and prints the original's summary block.
3. Evaluation is deterministic: the same model on the same validation tokens gives the same `val_bpb`.
4. A run that diverges (NaN, or loss above 100) fails fast with a non-zero exit.
5. `program.md` lets an agent run at least ten keep-or-discard cycles unattended, including a compile error and a crash.

### Quality

1. **Parity.** Model outputs, gradients, optimizer updates, token ids, batches and `val_bpb` match fixtures made from the PyTorch original, within tolerances written down per check.
2. **Reproducibility.** Two runs with the same seed give the same losses step for step on one device. The step at which the time budget stops them may differ.
3. **Legibility to the agent.** `train.cr` reads top to bottom like `train.py`: hyperparameters first, then model, optimizer, loop. Compile errors are the agent's main feedback, so the file avoids macros and metaprogramming beyond what tensomite's `Module` needs.
4. **Honest reporting.** Every deviation from the original (precision, scale, substituted attention) is stated where the number is reported.

## Out of scope

- Matching H100 throughput or the published `val_bpb`.
- Multi-GPU or distributed training.
- Windows.
- A general `autoresearch` framework. This is one recipe, ported.
- Library features. Anything the port needs from tensomite is an issue there, with this repo as its motivating use.

## Dependencies

- `tensomite` 0.8.4 or later. It provides optimizer groups and the `updates` hook, the elementwise helpers, per-token cross-entropy, weightless RMSNorm, the BPE pattern option, a `tiktoken` converter, a device memory high-water mark (`peak_live_bytes`, for `peak_vram_mb`), and collects modules held in a `Hash`.
- Open in tensomite's M14, and not blocking a scaled-down port: 64-bit sizes, fused windowed attention for training, BF16 compute, and fused cross-entropy.
- Crystal 1.22 or later; OpenBLAS on Linux.
- PyTorch, only to regenerate fixtures.

## Success

The project has succeeded when:

1. `make test` passes, including every parity spec.
2. A scaled-down profile trains inside its budget on at least one GPU and on CPU, and prints a `val_bpb`.
3. An agent completes at least ten unattended cycles from `program.md`, with the log committed as evidence.
4. Each gap found on the way is an issue in tensomite, closed or consciously deferred.