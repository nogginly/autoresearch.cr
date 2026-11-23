"""Writes the parity fixtures from karpathy/autoresearch's train.py.

Usage: python make_fixtures.py AUTORESEARCH_DIR FIXTURES_DIR

The model and optimizer code is train.py's own, read from AUTORESEARCH_DIR at
the pinned commit and run on the CPU in float32, with only the substitutions in
MODEL_PATCHES and OPTIMIZER_PATCHES. Writes to FIXTURES_DIR/model:

  weights.safetensors   the tiny model's parameters, named as in train.py and
                        train.cr, in PyTorch's layout (Linear weights [out, in])
  inputs.safetensors    idx and targets, I32 [B, T]; one target is -1 (ignored)
  expected.safetensors  logits [B*T, V], loss [1], per_token_loss [B*T], and
                        grad.<name> for every parameter, in PyTorch's layout
  expected.json         provenance, the tiny config, parameter names, and parameter counts and
                        FLOPs per token at depth 8

and to FIXTURES_DIR/optimizer, five MuonAdamW steps on the tiny model from fixed
gradients, with train.py's hyperparameters and schedules:

  tensors.safetensors   initial.<name>, grad.<k>.<name> for steps k = 1 to 5,
                        after1.<name> and after5.<name>, in PyTorch's layout
  expected.json         provenance, the hyperparameters, the optimizer's groups
                        as setup_optimizer makes them, and each step's inputs
                        to the schedules and the values they returned
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

# Optimizer steps: the progress through the time budget and the step number
# passed to the schedules, chosen to cover warm-down, weight-decay decay and
# momentum warm-up.
OPTIMIZER_STEPS = [(0.0, 0), (0.3, 75), (0.55, 150), (0.8, 225), (0.95, 300)]
GRAD_SCALE = 0.01

# Each patch is (original text, replacement, reason, occurrences); each original
# must occur exactly that many times.
MODEL_PATCHES = [
    (
        "y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)",
        "y = windowed_attention(q, k, v, window_size)",
        "FlashAttention 3 needs a Hopper GPU; scaled_dot_product_attention with the same "
        "causal sliding-window mask (key j visible from query i when 0 <= i - j <= window) "
        "computes the same function",
        1,
    ),
    (
        "        cos, sin = cos.bfloat16(), sin.bfloat16()\n",
        "",
        "rotary tables stay float32 (Decision G: F32 until tensomite computes in BF16)",
        1,
    ),
    (
        "        self.transformer.wte.to(dtype=torch.bfloat16)\n"
        "        for ve in self.value_embeds.values():\n"
        "            ve.to(dtype=torch.bfloat16)\n",
        "",
        "embeddings stay float32 (Decision G)",
        1,
    ),
]

OPTIMIZER_PATCHES = [
    (
        "@torch.compile(dynamic=False, fullgraph=True)\n",
        "",
        "the fused steps run eagerly, not through torch.compile, so the CPU needs no compiler "
        "toolchain and the arithmetic is as written",
        2,
    ),
    (
        "    X = g.bfloat16()\n",
        "    X = g\n",
        "Newton-Schulz runs in float32, not BF16 (Decision G)",
        1,
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
    """Returns train.py's model and optimizer code, patched, as a namespace; its
    hyperparameters; build_model_config; and the three schedule functions."""
    train_path = os.path.join(source_dir, "train.py")
    prepare_path = os.path.join(source_dir, "prepare.py")
    check_sha256(train_path, TRAIN_SHA256)
    check_sha256(prepare_path, PREPARE_SHA256)
    train = open(train_path).read()
    prepare = open(prepare_path).read()

    start = train.index("@dataclass\nclass GPTConfig")
    end = train.index("# ---------------------------------------------------------------------------\n# Hyperparameters")
    code = train[start:end]
    for old, new, _, occurrences in MODEL_PATCHES + OPTIMIZER_PATCHES:
        count = code.count(old)
        if count != occurrences:
            sys.exit(f"patch target occurs {count} times, not {occurrences}: {old!r}")
        code = code.replace(old, new)

    namespace = dict(torch=torch, nn=nn, F=F, math=math, dataclass=dataclass, asdict=asdict,
                     windowed_attention=windowed_attention, __name__="autoresearch_train")
    exec(compile(code, train_path, "exec"), namespace)

    def constant(source, name):
        return eval(re.search(rf"^{name} = (.+?)\s*(#.*)?$", source, re.M).group(1))

    hyperparameters = {
        name: constant(train, name)
        for name in ["ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN", "EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR",
                     "SCALAR_LR", "WEIGHT_DECAY", "ADAM_BETAS", "WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC"]
    }
    constants = dict(hyperparameters, MAX_SEQ_LEN=constant(prepare, "MAX_SEQ_LEN"), vocab_size=constant(prepare, "VOCAB_SIZE"))
    extra = dict(namespace, **constants)
    for start, end in [("def build_model_config(depth):", "config = build_model_config(DEPTH)"),
                       ("def get_lr_multiplier(progress):", "# ---------------------------------------------------------------------------\n# Training loop")]:
        exec(compile(train[train.index(start):train.index(end)], train_path, "exec"), extra)
    schedules = {name: extra[name] for name in ["get_lr_multiplier", "get_muon_momentum", "get_weight_decay"]}
    return namespace, hyperparameters, extra["build_model_config"], schedules


def make_model(ns, build_model_config, out_dir):
    """Writes the model fixtures and returns the tiny model, its parameters as they were saved."""
    os.makedirs(out_dir, exist_ok=True)
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

    # 2. One batch, the loss, the per-token losses and the gradients.
    idx = torch.randint(0, config.vocab_size, (BATCH, config.sequence_len))
    targets = torch.randint(0, config.vocab_size, (BATCH, config.sequence_len))
    targets[0, 3] = -1
    logits = model(idx)
    per_token = model(idx, targets, reduction="none")
    loss = model(idx, targets)
    loss.backward()

    weights = {n: p.detach().contiguous() for n, p in model.named_parameters()}
    expected = {
        "logits": logits.detach().reshape(-1, config.vocab_size).contiguous(),
        "loss": loss.detach().reshape(1),
        "per_token_loss": per_token.detach().reshape(-1).contiguous(),
    }
    expected.update({f"grad.{n}": p.grad.contiguous() for n, p in model.named_parameters()})
    save_file(weights, os.path.join(out_dir, "weights.safetensors"))
    save_file({"idx": idx.to(torch.int32), "targets": targets.to(torch.int32)}, os.path.join(out_dir, "inputs.safetensors"))
    save_file(expected, os.path.join(out_dir, "expected.safetensors"))

    # 3. Counts and FLOPs at depth 8, on the meta device so nothing is allocated.
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
        value_embed_layers=sorted(int(key) for key in model.value_embeds.keys()),
        substitutions=[reason for _, _, reason, _ in MODEL_PATCHES] + OTHER_SUBSTITUTIONS,
        names=[name for name, _ in model.named_parameters()],
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
    return model


def make_optimizer(model, hyperparameters, schedules, out_dir):
    """Writes the optimizer fixtures: five steps of train.py's MuonAdamW on *model* from fixed gradients."""
    os.makedirs(out_dir, exist_ok=True)
    h = hyperparameters
    params = dict(model.named_parameters())
    tensors = {f"initial.{n}": p.detach().clone().contiguous() for n, p in params.items()}

    # 1. The optimizer as train.py sets it up, with its hyperparameters.
    optimizer = model.setup_optimizer(
        unembedding_lr=h["UNEMBEDDING_LR"], embedding_lr=h["EMBEDDING_LR"], scalar_lr=h["SCALAR_LR"],
        adam_betas=h["ADAM_BETAS"], matrix_lr=h["MATRIX_LR"], weight_decay=h["WEIGHT_DECAY"],
    )
    name_of = {id(p): n for n, p in params.items()}
    groups = []
    for group in optimizer.param_groups:
        entry = {k: v for k, v in group.items() if k != "params"}
        entry["params"] = [name_of[id(p)] for p in group["params"]]
        if "betas" in entry:
            entry["betas"] = list(entry["betas"])
        groups.append(entry)

    # 2. Five steps, each from a fixed gradient, with the schedules applied as train.py's loop does.
    generator = torch.Generator().manual_seed(SEED + 1)
    steps = []
    for k, (progress, step) in enumerate(OPTIMIZER_STEPS, start=1):
        lrm = schedules["get_lr_multiplier"](progress)
        muon_momentum = schedules["get_muon_momentum"](step)
        muon_weight_decay = schedules["get_weight_decay"](progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        for n, p in params.items():
            p.grad = GRAD_SCALE * torch.randn(p.shape, generator=generator)
            tensors[f"grad.{k}.{n}"] = p.grad.clone().contiguous()
        optimizer.step()
        if k in (1, 5):
            tensors.update({f"after{k}.{n}": p.detach().clone().contiguous() for n, p in params.items()})
        steps.append(dict(progress=progress, step=step, lr_multiplier=lrm, muon_momentum=muon_momentum,
                          muon_weight_decay=muon_weight_decay))

    save_file(tensors, os.path.join(out_dir, "tensors.safetensors"))
    meta = dict(
        source="https://github.com/karpathy/autoresearch",
        commit=COMMIT,
        torch=torch.__version__,
        start="the tiny model of ../model, its weights as in ../model/weights.safetensors",
        grad_scale=GRAD_SCALE,
        hyperparameters={k: list(v) if isinstance(v, tuple) else v for k, v in h.items()},
        substitutions=[reason for _, _, reason, _ in OPTIMIZER_PATCHES],
        groups=groups,
        steps=steps,
    )
    with open(os.path.join(out_dir, "expected.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"{len(steps)} optimizer steps; wrote {out_dir}")


def main():
    source_dir, fixtures_dir = sys.argv[1], sys.argv[2]
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    ns, hyperparameters, build_model_config, schedules = load_original(source_dir)
    model = make_model(ns, build_model_config, os.path.join(fixtures_dir, "model"))
    make_optimizer(model, hyperparameters, schedules, os.path.join(fixtures_dir, "optimizer"))


if __name__ == "__main__":
    main()
