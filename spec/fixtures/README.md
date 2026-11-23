# Fixtures

Reference outputs from the PyTorch original, committed so that specs and CI need no Python. `make fixtures` regenerates all of them (see DEVELOPMENT.md, "Fixtures"); the output is byte-identical from run to run.

## Source

- [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) at commit `228791fb499afffb54b46200aca536f79142f117`. The script checks the SHA-256 of `train.py` and `prepare.py` before using them.
- PyTorch 2.9.1 on the CPU, in float32, one thread, deterministic algorithms (`scripts/fixtures/requirements.txt` pins every package).
- The model code is `train.py`'s own, from `@dataclass class GPTConfig` up to the hyperparameters, executed as written except for the patches below. `build_model_config` and its constants are read from `train.py` and `prepare.py` too.

## Substitutions

Each is listed, with its reason, in `model/expected.json` under `substitutions`.

| Original | Fixture | Why |
|---|---|---|
| `fa3.flash_attn_func(q, k, v, causal=True, window_size=w)` | `scaled_dot_product_attention` with a mask: key `j` visible from query `i` when `0 <= i - j <= w[0]` | FlashAttention 3 needs a Hopper GPU. The mask is FlashAttention's documented sliding window, inclusive at both ends |
| Rotary tables cast to BF16 | float32 | Decision G |
| `wte` and value embeddings cast to BF16 | float32 | Decision G |
| `torch.autocast` to BF16 | none | Decision G |
| Zero-initialised `c_proj`, `mlp.c_proj`, `ve_gate`; `lm_head` normal(0.001); scalars 1 and 0.1 | `c_proj`s and gates uniform(±s/2); `lm_head` normal(1); scalars with per-layer noise | With the original's initial values most gradients are zero, the logits are near 0 and the loss is ln V, so a broken branch or softcap would go unnoticed |

## `model/`

The tiny model: 4 layers, width 64, 2 heads, vocabulary 128, sequence 16, batch 2, window pattern `SLS`. The pattern gives windows `[8, 16, 8, 16]`: it exercises the pattern's wrap-around and the rule that the last layer is always long, which `SSSL` at 4 layers would not.

| File | Contents |
|---|---|
| `weights.safetensors` | every parameter, named as in `train.cr`, in PyTorch's layout (`Linear` weights `[out, in]`) |
| `inputs.safetensors` | `idx` and `targets`, `I32` `[2, 16]`; `targets[0, 3]` is -1, which the loss ignores |
| `expected.safetensors` | `logits` `[32, 128]`, `loss` `[1]`, `per_token_loss` `[32]`, and `grad.<name>` for every parameter, in PyTorch's layout |
| `expected.json` | provenance, the tiny config, the window sizes, the substitutions, the name map, and `num_scaling_params` and `estimate_flops` at depth 8 |

**Names.** `train.py` keys value embeddings by layer (`value_embeds.1`, `value_embeds.3`); `train.cr` keeps them in a list in layer order (`value_embeds.0`, `value_embeds.1`), because a tensomite `Module` cannot hold a keyed collection ([tensomite#88](https://github.com/nogginly/tensomite.cr/issues/88)). `train.py`'s `transformer.` prefix is dropped. `names` in `expected.json` maps each original name to the port's.
