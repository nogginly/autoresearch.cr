# Fixtures

Reference outputs from the PyTorch original, committed so that specs and CI need no Python. `make fixtures` regenerates all of them (see DEVELOPMENT.md, "Fixtures"); the output is byte-identical from run to run.

## Source

- [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) at commit `228791fb499afffb54b46200aca536f79142f117`. The script checks the SHA-256 of `train.py` and `prepare.py` before using them.
- PyTorch 2.9.1 on the CPU, in float32, one thread, deterministic algorithms (`scripts/fixtures/requirements.txt` pins every package).
- The model and optimizer code is `train.py`'s own, from `@dataclass class GPTConfig` up to the hyperparameters, executed as written except for the patches below. `build_model_config`, the three schedule functions and the hyperparameters they use are read from `train.py` and `prepare.py` too. One script, `scripts/fixtures/make_fixtures.py`, writes every fixture.

## Substitutions

Each is listed, with its reason, under `substitutions` in `model/expected.json` (the first five) or `optimizer/expected.json` (the last two).

| Original | Fixture | Why |
|---|---|---|
| `fa3.flash_attn_func(q, k, v, causal=True, window_size=w)` | `scaled_dot_product_attention` with a mask: key `j` visible from query `i` when `0 <= i - j <= w[0]` | FlashAttention 3 needs a Hopper GPU. The mask is FlashAttention's documented sliding window, inclusive at both ends |
| Rotary tables cast to BF16 | float32 | Decision G |
| `wte` and value embeddings cast to BF16 | float32 | Decision G |
| `torch.autocast` to BF16 | none | Decision G |
| Zero-initialised `c_proj`, `mlp.c_proj`, `ve_gate`; `lm_head` normal(0.001); scalars 1 and 0.1 | `c_proj`s and gates uniform(±s/2); `lm_head` normal(1); scalars with per-layer noise | With the original's initial values most gradients are zero, the logits are near 0 and the loss is ln V, so a broken branch or softcap would go unnoticed |
| `@torch.compile` on the fused AdamW and Muon steps | run eagerly | The CPU needs no compiler toolchain, and the arithmetic is as written |
| Newton–Schulz on `g.bfloat16()` | on `g`, float32 | Decision G |

## `model/`

The tiny model: 4 layers, width 64, 2 heads, vocabulary 128, sequence 16, batch 2, window pattern `SLS`. The pattern gives windows `[8, 16, 8, 16]`: it exercises the pattern's wrap-around and the rule that the last layer is always long, which `SSSL` at 4 layers would not.

| File | Contents |
|---|---|
| `weights.safetensors` | every parameter, named as in `train.py` and `train.cr`, in PyTorch's layout (`Linear` weights `[out, in]`) |
| `inputs.safetensors` | `idx` and `targets`, `I32` `[2, 16]`; `targets[0, 3]` is -1, which the loss ignores |
| `expected.safetensors` | `logits` `[32, 128]`, `loss` `[1]`, `per_token_loss` `[32]`, and `grad.<name>` for every parameter, in PyTorch's layout |
| `expected.json` | provenance, the tiny config, the window sizes, the substitutions, the parameter names, and `num_scaling_params` and `estimate_flops` at depth 8 |

**Names.** Every parameter has `train.py`'s name, such as `transformer.h.0.attn.c_q.weight` and `value_embeds.1.weight`, which `train.cr` shares; `names` in `expected.json` lists them.

## `optimizer/`

Five steps of `train.py`'s `MuonAdamW` on the tiny model, from its weights in `model/weights.safetensors`, set up by `setup_optimizer` with `train.py`'s hyperparameters. Each step:

1. sets every group's learning rate, and Muon's momentum and weight decay, from the schedules, as `train.py`'s loop does;
2. sets every gradient to fixed noise, `0.01 × normal(0, 1)` from a seeded generator;
3. calls `optimizer.step()`.

The schedules' inputs, `(progress, step)`, are `(0, 0)`, `(0.3, 75)`, `(0.55, 150)`, `(0.8, 225)` and `(0.95, 300)`. They cover the plateau and the warm-down of the learning rate, the decay of weight decay, and momentum's warm-up from 0.85 to 0.95. The tiny model's matrices give Muon square (`c_q`), tall (`c_fc`, 256 × 64) and wide (`mlp.c_proj`, `ve_gate`) groups.

| File | Contents |
|---|---|
| `tensors.safetensors` | `initial.<name>`; `grad.<k>.<name>` for steps 1 to 5; `after1.<name>` and `after5.<name>`; all in PyTorch's layout |
| `expected.json` | provenance, the hyperparameters, the groups as `setup_optimizer` makes them (before any step), each step's schedule inputs and outputs, and the substitutions |

## `data/`

A small dataset in the climbmix shards' layout, and what `prepare.py` makes of it. It uses `prepare.py`'s own `train_tokenizer`, `Tokenizer`, `get_token_bytes` and `make_dataloader`, with its directories pointed at a scratch cache. The dataloader is patched to stay on the CPU: its host buffer is not pinned, and its batches are not moved to CUDA.

**Documents.** There are 1,700 windows of the original's own `README.md`, `program.md`, `prepare.py` and `train.py` at the pinned commit (MIT licence). Their lengths are log-uniform between 20 and 3,000 characters, and three run to 12,000, which exercises the 10,000-character cut. Every tenth document gains a Unicode sample: CJK, emoji, combining marks, no-break and ideographic spaces, U+2028 and U+0085, Arabic digits, contractions, CRLF and look-alikes of the special tokens.

**Shards.** `shard_00000` and `shard_00001` (700 documents each) are the training split, and `shard_06542` (300) is the validation split. They are written by pyarrow as the real shards are: ZSTD, PLAIN, v1 pages, no dictionary, no statistics. Row groups are 300 documents, so the dataloader's batches of 128 end short (128, 128, 44).

| File | Contents |
|---|---|
| `parquet/` | the three shards |
| `expected.json` | provenance; row groups; the tokenizer's 5,243 ranked tokens (hex bytes), vocabulary size and BOS id; each token's byte length; each document's ids, BOS first, as the SHA-256 of little-endian int32s, with the first five in full; each loader's batch, sequence, steps and epochs |
| `batches.safetensors` | `train.<k>.inputs` and `.targets` for 120 batches of 4 × 512, and `val.<k>.*` for 8; I32 |

The training split is shorter than 120 batches of 2,052 tokens, so they cross into epoch 2. The validation split is smaller than the 1,000-document buffer, so its first batch already reports epoch 4. On so small a corpus, training runs out of pairs after 4,987 merges, which exercises the trainer's early stop.

