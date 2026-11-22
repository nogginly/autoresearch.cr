"""Writes the model parity fixtures from karpathy/autoresearch's train.py.

Usage: python make_model_fixtures.py AUTORESEARCH_DIR OUT_DIR

The model code is train.py's own, read from AUTORESEARCH_DIR at the pinned
commit and run on the CPU in float32, with only the substitutions in PATCHES.
Writes to OUT_DIR:

  weights.safetensors   the tiny model's parameters, named as in train.cr,
                        in PyTorch's layout (Linear weights [out, in])
  inputs.safetensors    idx and targets, I32 [B, T]; one target is -1 (ignored)
  expected.safetensors  logits [B*T, V], loss [1], per_token_loss [B*T], and
                        grad.<name> for every parameter, in PyTorch's layout
  expected.json         provenance, the tiny config, and parameter counts and
                        FLOPs per token at depth 8
"""

import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

COMMIT = "228791fb499afffb54b46200aca536f79142f117"
TRAIN_SHA256 = "2954175f4ac42ad65164aef40910ef953789abcd05a5cc886ac9ba5a00814414"
PREPARE_SHA256 = "4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150"

TINY = dict(sequence_len=16, vocab_size=128, n_layer=4, n_head=2, n_kv_head=2, n_embd=64, window_pattern="SLS")
BATCH = 2
SEED = 42

# Each patch is (original text, replacement, reason); each original must occur exactly once.
PATCHES = [
    (
        "y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)",
        "y = windowed_attention(q, k, v, window_size)",
        "FlashAttention 3 needs a Hopper GPU; scaled_dot_product_attention with the same "
        "causal sliding-window mask (key j visible from query i when 0 <= i - j <= window) "
        "computes the same function",
    ),
    (
        "        cos, sin = cos.bfloat16(), sin.bfloat16()\n",
        "",
        "rotary tables stay float32 (Decision G: F32 until tensomite computes in BF16)",
    ),
    (
        "        self.transformer.wte.to(dtype=torch.bfloat16)\n"
        "        for ve in self.value_embeds.values():\n"
        "            ve.to(dtype=torch.bfloat16)\n",
        "",
        "embeddings stay float32 (Decision G)",
    ),
]

OTHER_SUBSTITUTIONS = [
    "No torch.autocast: every operation runs in float32 on the CPU.",
    "Weights that train.py initialises to zero (c_proj, mlp.c_proj, ve_gate) are drawn from "
    "uniform(-s/2, s/2), lm_head is drawn from normal(0, 1) so that logits reach the softcap's "
    "non-linear range, and resid_lambdas and x0_lambdas get per-layer noise, so that every "
    "gradient and every per-layer scalar is exercised.",
]


def windowed_attention(q, k, v, window_size):
    """FlashAttention's (B, T, H, D) causal sliding-window attention, by scaled_dot_product_attention."""
    t = q.size(1)
    i = torch.arange(t)[:, None]
    j = torch.arange(t)[None, :]
    allowed = (j <= i) & (i - j <= window_size[0])
    y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed)
    return y.transpose(1, 2)


def check_sha256(path, expected):
    with open(path, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if actual != expected:
        sys.exit(f"{path}: sha256 {actual}, expected {expected} (commit {COMMIT})")


def load_original(source_dir):
    """Returns train.py's model code, patched, as a namespace, and prepare.py's and train.py's constants."""
    train_path = os.path.join(source_dir, "train.py")
    prepare_path = os.path.join(source_dir, "prepare.py")
    check_sha256(train_path, TRAIN_SHA256)
    check_sha256(prepare_path, PREPARE_SHA256)
    train = open(train_path).read()
    prepare = open(prepare_path).read()

    start = train.index("@dataclass\nclass GPTConfig")
    end = train.index("# ---------------------------------------------------------------------------\n# Hyperparameters")
    code = train[start:end]
    for old, new, _ in PATCHES:
        count = code.count(old)
        if count != 1:
            sys.exit(f"patch target occurs {count} times, not once: {old!r}")
        code = code.replace(old, new)

    namespace = dict(torch=torch, nn=nn, F=F, math=math, dataclass=dataclass, asdict=asdict,
                     windowed_attention=windowed_attention, __name__="autoresearch_train")
    exec(compile(code, train_path, "exec"), namespace)

    def constant(source, name):
        return eval(re.search(rf"^{name} = (.+?)\s*(#.*)?$", source, re.M).group(1))

    constants = dict(
        ASPECT_RATIO=constant(train, "ASPECT_RATIO"),
        HEAD_DIM=constant(train, "HEAD_DIM"),
        WINDOW_PATTERN=constant(train, "WINDOW_PATTERN"),
        MAX_SEQ_LEN=constant(prepare, "MAX_SEQ_LEN"),
        vocab_size=constant(prepare, "VOCAB_SIZE"),
    )
    build_start = train.index("def build_model_config(depth):")
    build_end = train.index("config = build_model_config(DEPTH)")
    build_namespace = dict(namespace, **constants)
    exec(compile(train[build_start:build_end], train_path, "exec"), build_namespace)
    return namespace, build_namespace["build_model_config"]


def port_name(name):
    """Maps a train.py parameter name to train.cr's."""
    name = re.sub(r"^transformer\.", "", name)
    return name


def main():
    source_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    ns, build_model_config = load_original(source_dir)
    GPT, GPTConfig = ns["GPT"], ns["GPTConfig"]

    # 1. The tiny model, initialised as train.py does, then perturbed (see OTHER_SUBSTITUTIONS).
    torch.manual_seed(SEED)
    config = GPTConfig(**TINY)
    model = GPT(config)
    model.init_weights()
    s = 3**0.5 * config.n_embd**-0.5
    with torch.no_grad():
        for block in model.transformer.h:
            torch.nn.init.uniform_(block.attn.c_proj.weight, -s / 2, s / 2)
            torch.nn.init.uniform_(block.mlp.c_proj.weight, -s / 2, s / 2)
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, -s / 2, s / 2)
        torch.nn.init.normal_(model.lm_head.weight, mean=0.0, std=1.0)
        model.resid_lambdas.add_(0.1 * torch.randn(config.n_layer))
        model.x0_lambdas.add_(0.05 * torch.randn(config.n_layer))

    # 2. Value embeddings are keyed by layer in train.py and listed in layer order in train.cr.
    ve_layers = sorted(int(key) for key in model.value_embeds.keys())
    names = {}
    for name, _ in model.named_parameters():
        m = re.match(r"^value_embeds\.(\d+)\.weight$", name)
        names[name] = f"value_embeds.{ve_layers.index(int(m.group(1)))}.weight" if m else port_name(name)

    # 3. One batch, the loss, the per-token losses and the gradients.
    idx = torch.randint(0, config.vocab_size, (BATCH, config.sequence_len))
    targets = torch.randint(0, config.vocab_size, (BATCH, config.sequence_len))
    targets[0, 3] = -1
    logits = model(idx)
    per_token = model(idx, targets, reduction="none")
    loss = model(idx, targets)
    loss.backward()

    weights = {names[n]: p.detach().contiguous() for n, p in model.named_parameters()}
    expected = {
        "logits": logits.detach().reshape(-1, config.vocab_size).contiguous(),
        "loss": loss.detach().reshape(1),
        "per_token_loss": per_token.detach().reshape(-1).contiguous(),
    }
    expected.update({f"grad.{names[n]}": p.grad.contiguous() for n, p in model.named_parameters()})
    save_file(weights, os.path.join(out_dir, "weights.safetensors"))
    save_file({"idx": idx.to(torch.int32), "targets": targets.to(torch.int32)}, os.path.join(out_dir, "inputs.safetensors"))
    save_file(expected, os.path.join(out_dir, "expected.safetensors"))

    # 4. Counts and FLOPs at depth 8, on the meta device so nothing is allocated.
    with torch.device("meta"):
        reference = GPT(build_model_config(8))
    meta = dict(
        source="https://github.com/karpathy/autoresearch",
        commit=COMMIT,
        torch=torch.__version__,
        seed=SEED,
        batch=BATCH,
        tiny=asdict(config),
        window_sizes=[w[0] for w in model.window_sizes],
        value_embed_layers=ve_layers,
        substitutions=[reason for _, _, reason in PATCHES] + OTHER_SUBSTITUTIONS,
        names={original: names[original] for original in names},
        depth8=dict(
            config=asdict(reference.config),
            num_scaling_params=reference.num_scaling_params(),
            estimate_flops=reference.estimate_flops(),
        ),
        loss=loss.item(),
    )
    with open(os.path.join(out_dir, "expected.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"loss {loss.item():.6f}; wrote {out_dir}")


if __name__ == "__main__":
    main()
