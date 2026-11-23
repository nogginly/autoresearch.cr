# Autoresearch pretraining script: the one file the agent edits.
# Build with `make build` and run `bin/train`.

require "tensomite"
require "tensomite/blas"
require "./src/autoresearch/constants"

include Tensomite

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO   =  64    # model_dim = depth * ASPECT_RATIO
HEAD_DIM       = 128    # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L = full, S = half context

# Optimization
EMBEDDING_LR   =   0.6       # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004       # learning rate for lm_head (Adam)
MATRIX_LR      =  0.04       # learning rate for matrix parameters (Muon)
SCALAR_LR      =   0.5       # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY   =   0.2       # cautious weight decay for Muon
ADAM_BETAS     = {0.8, 0.95} # Adam beta1, beta2
WARMUP_RATIO   = 0.0         # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5         # fraction of time budget for LR warmdown
FINAL_LR_FRAC  = 0.0         # final LR as fraction of initial

# Model size
DEPTH = 8 # number of transformer layers

# ---------------------------------------------------------------------------
# GPT model
# ---------------------------------------------------------------------------

# `F.rms_norm`'s epsilon for float32.
RMS_EPS          = Float32::EPSILON.to_f64
SOFTCAP          = 15.0
VE_GATE_CHANNELS =   32

record GPTConfig,
  sequence_len : Int32 = 2048,
  vocab_size : Int32 = 32768,
  n_layer : Int32 = 12,
  n_head : Int32 = 6,
  n_kv_head : Int32 = 6,
  n_embd : Int32 = 768,
  window_pattern : String = "SSSL"

# Returns *x* RMS-normalised over its last axis, without a weight.
def norm(x : Tensor) : Tensor
  x.rmsnorm(RMS_EPS)
end

# Returns true if *layer* has a value embedding: alternating layers, the last always included.
def has_ve(layer : Int32, n_layer : Int32) : Bool
  layer % 2 == (n_layer - 1) % 2
end

# Returns a `[t, t]` additive mask: 0 where query `i` may see key `j`
# (`j <= i` and `i - j <= window`, FlashAttention's causal sliding window), -∞ elsewhere.
def window_mask(t : Int32, window : Int32, device : Device) : Tensor
  data = Array(Float32).new(t * t) do |index|
    i, j = index.divmod(t)
    j <= i && i - j <= window ? 0f32 : -Float32::INFINITY
  end
  Tensor.from(data, [t, t], device)
end

class CausalSelfAttention < Tensomite::Module
  getter n_head : Int32
  getter head_dim : Int32
  @c_q : Linear
  @c_k : Linear
  @c_v : Linear
  @c_proj : Linear
  @ve_gate : Linear?

  def initialize(config : GPTConfig, layer : Int32, build : Builder)
    unless config.n_kv_head == config.n_head
      raise ArgumentError.new("n_kv_head #{config.n_kv_head} != n_head #{config.n_head}: grouped-query attention is not ported")
    end
    @n_head = config.n_head
    @head_dim = config.n_embd // config.n_head
    raise ArgumentError.new("n_embd #{config.n_embd} is not a multiple of n_head #{config.n_head}") unless @head_dim * @n_head == config.n_embd
    @c_q = build.linear(config.n_embd, @n_head * @head_dim, bias: false)
    @c_k = build.linear(config.n_embd, @n_head * @head_dim, bias: false)
    @c_v = build.linear(config.n_embd, @n_head * @head_dim, bias: false)
    @c_proj = build.linear(config.n_embd, config.n_embd, bias: false)
    @ve_gate = has_ve(layer, config.n_layer) ? build.linear(VE_GATE_CHANNELS, @n_head, bias: false) : nil
  end

  # Returns the attention output for *x*, `[b * t, n_embd]`, mixing in the value embedding *ve* when given.
  def forward(x : Tensor, ve : Tensor?, cos_sin : {Tensor, Tensor}, mask : Tensor, b : Int32, t : Int32) : Tensor
    h, d = @n_head, @head_dim
    q = @c_q.call(x).reshape([b, t, h, d])
    k = @c_k.call(x).reshape([b, t, h, d])
    v = @c_v.call(x).reshape([b, t, h, d])

    # Value residual (ResFormer): mix in the value embedding with an input-dependent gate per head.
    if ve && (gate = @ve_gate)
      g = gate.call(x.slice(-1, 0, VE_GATE_CHANNELS)).sigmoid * 2.0
      v = v + g.reshape([b, t, h, 1]) * ve.reshape([b, t, h, d])
    end

    cos, sin = cos_sin
    q = norm(q.rope(cos, sin)).permute([0, 2, 1, 3])
    k = norm(k.rope(cos, sin)).permute([0, 2, 1, 3])
    v = v.permute([0, 2, 1, 3])

    probs = q.matmul(k.transpose).softmax(scale: 1.0 / Math.sqrt(d.to_f), mask: mask)
    y = probs.matmul(v).permute([0, 2, 1, 3]).contiguous.reshape([b * t, h * d])
    @c_proj.call(y)
  end

  # Fills the weights as `GPT#init_weights` describes.
  def init_weights(s : Float64, rng : Random) : Nil
    {@c_q, @c_k, @c_v}.each { |linear| fill(linear.weight, Init.uniform(s), rng) }
    fill(@c_proj.weight, Init.constant(0), rng)
    @ve_gate.try { |gate| fill(gate.weight, Init.constant(0), rng) }
  end
end

class MLP < Tensomite::Module
  @c_fc : Linear
  @c_proj : Linear

  def initialize(config : GPTConfig, build : Builder)
    @c_fc = build.linear(config.n_embd, 4 * config.n_embd, bias: false)
    @c_proj = build.linear(4 * config.n_embd, config.n_embd, bias: false)
  end

  # Returns `c_proj(relu(c_fc(x))²)`.
  def forward(x : Tensor) : Tensor
    @c_proj.call(@c_fc.call(x).relu.square)
  end

  # Fills the weights as `GPT#init_weights` describes.
  def init_weights(s : Float64, rng : Random) : Nil
    fill(@c_fc.weight, Init.uniform(s), rng)
    fill(@c_proj.weight, Init.constant(0), rng)
  end
end

class Block < Tensomite::Module
  getter attn : CausalSelfAttention
  getter mlp : MLP

  def initialize(config : GPTConfig, layer : Int32, build : Builder)
    @attn = CausalSelfAttention.new(config, layer, build)
    @mlp = MLP.new(config, build)
  end

  # Returns *x* after the attention and MLP residual branches, each on the normalised input.
  def forward(x : Tensor, ve : Tensor?, cos_sin : {Tensor, Tensor}, mask : Tensor, b : Int32, t : Int32) : Tensor
    x = x + @attn.call(norm(x), ve, cos_sin, mask, b, t)
    x + @mlp.call(norm(x))
  end
end

# The token embedding and the blocks, grouped as in train.py so that parameter names match.
class Transformer < Tensomite::Module
  getter wte : Embedding
  getter h : Array(Block)

  def initialize(config : GPTConfig, build : Builder)
    @wte = build.embedding(config.vocab_size, config.n_embd)
    @h = Array(Block).new(config.n_layer) { |i| Block.new(config, i, build) }
  end
end

# The model of `train.py`. Built with placeholder weights, which take no memory:
# call `#init_weights` to train from scratch, or `Module#load` to load a checkpoint.
class GPT < Tensomite::Module
  getter config : GPTConfig
  getter window_sizes : Array(Int32)
  @transformer : Transformer
  @lm_head : Linear
  @resid_lambdas : Parameter
  @x0_lambdas : Parameter
  @value_embeds : Hash(String, Embedding)

  def initialize(@config : GPTConfig, @device : Device)
    build = Builder.new(device: device, init: Init.placeholder)
    @window_sizes = compute_window_sizes(config)
    @transformer = Transformer.new(config, build)
    @lm_head = build.linear(config.n_embd, config.vocab_size, bias: false)
    @resid_lambdas = build.parameter([config.n_layer], decay: false)
    @x0_lambdas = build.parameter([config.n_layer], decay: false)
    # Value embeddings, keyed by the layer that has one.
    head_dim = config.n_embd // config.n_head
    kv_dim = config.n_kv_head * head_dim
    @value_embeds = {} of String => Embedding
    (0...config.n_layer).each do |i|
      @value_embeds[i.to_s] = build.embedding(config.vocab_size, kv_dim) if has_ve(i, config.n_layer)
    end
    # Rotary embeddings. Tensomite rotates as `x·cos + rotate_half(x)·sin`; train.py
    # as `x·cos − rotate_half(x)·sin`, so `#forward` passes `−sin`.
    @rope = RotaryEmbedding.new(head_dim, context: config.sequence_len * 10, device: device)
    @masks = {} of {Int32, Int32} => Tensor
  end

  # Fills every weight from *rng*, as train.py's `init_weights` does: embedding
  # normal(1), head normal(0.001), matrices and value embeddings uniform ±√3/√n_embd,
  # output projections and value-embedding gates zero, `resid_lambdas` 1 and `x0_lambdas` 0.1.
  def init_weights(rng : Random) : Nil
    s = Math.sqrt(3.0) / Math.sqrt(config.n_embd.to_f)
    fill(@transformer.wte.weight, Init.normal(1.0), rng)
    fill(@lm_head.weight, Init.normal(0.001), rng)
    @transformer.h.each do |block|
      block.attn.init_weights(s, rng)
      block.mlp.init_weights(s, rng)
    end
    fill(@resid_lambdas, Init.constant(1.0), rng)
    fill(@x0_lambdas, Init.constant(0.1), rng)
    @value_embeds.each_value { |embedding| fill(embedding.weight, Init.uniform(s), rng) }
  end

  # Returns each layer's attention window: half the context for `S`, all of it
  # for `L`, and all of it for the last layer whatever the pattern says.
  def compute_window_sizes(config : GPTConfig) : Array(Int32)
    pattern = config.window_pattern.upcase
    raise ArgumentError.new("window pattern #{pattern} must hold only S and L") unless pattern.each_char.all?(&.in?('S', 'L'))
    long_window = config.sequence_len
    short_window = long_window // 2
    sizes = Array(Int32).new(config.n_layer) { |i| pattern[i % pattern.size] == 'L' ? long_window : short_window }
    sizes[-1] = long_window
    sizes
  end

  # Returns the estimated FLOPs per token, forward and backward, as train.py counts them.
  def estimate_flops : Int64
    nparams = count(params)
    nparams_exclude = count(params_named("transformer.wte.") + params_named("value_embeds.") + [@resid_lambdas, @x0_lambdas])
    h = config.n_head.to_i64
    q = (config.n_embd // config.n_head).to_i64
    t = config.sequence_len
    attn_flops = window_sizes.sum(0i64) { |window| 12i64 * h * q * (window < 0 ? t : Math.min(window, t)) }
    6i64 * (nparams - nparams_exclude) + attn_flops
  end

  # Returns the parameter counts by group, as train.py's `num_scaling_params`.
  def num_scaling_params : Hash(String, Int64)
    counts = {
      "wte"                  => count(params_named("transformer.wte.")),
      "value_embeds"         => count(params_named("value_embeds.")),
      "lm_head"              => count(params_named("lm_head.")),
      "transformer_matrices" => count(params_named("transformer.h.")),
      "scalars"              => count([@resid_lambdas, @x0_lambdas]),
    }
    counts["total"] = counts.values.sum
    counts
  end

  # Returns a `MuonAdamW` over every parameter, grouped as train.py groups them:
  # Adam for the head, the token embedding, the value embeddings and the two
  # scalar vectors, with the head's and embeddings' rates scaled by
  # `(n_embd / 768)^-½`; Muon for the blocks' matrices, one group per shape.
  def setup_optimizer(unembedding_lr = 0.004, embedding_lr = 0.2, matrix_lr = 0.02,
                      weight_decay = 0.0, adam_betas = {0.8, 0.95}, scalar_lr = 0.5) : MuonAdamW
    named = named_params.to_a
    indices = ->(prefix : String) { (0...named.size).select { |i| named[i][0].starts_with?(prefix) } }
    matrix_params = indices.call("transformer.h.")
    dmodel_lr_scale = (config.n_embd / 768) ** -0.5
    groups = [
      AdamWGroup.new(indices.call("lm_head."), unembedding_lr * dmodel_lr_scale, adam_betas, 1e-10, 0.0),
      AdamWGroup.new(indices.call("transformer.wte."), embedding_lr * dmodel_lr_scale, adam_betas, 1e-10, 0.0),
      AdamWGroup.new(indices.call("value_embeds."), embedding_lr * dmodel_lr_scale, adam_betas, 1e-10, 0.0),
      AdamWGroup.new(indices.call("resid_lambdas"), scalar_lr * 0.01, adam_betas, 1e-10, 0.0),
      AdamWGroup.new(indices.call("x0_lambdas"), scalar_lr, {0.96, 0.95}, 1e-10, 0.0),
    ] of ParamGroup
    # train.py orders shapes as [out, in], PyTorch's layout; tensomite stores [in, out].
    shapes = matrix_params.map { |i| named[i][1].shape.reverse }.uniq!.sort!
    shapes.each do |shape|
      members = matrix_params.select { |i| named[i][1].shape.reverse == shape }
      groups << MuonGroup.new(members, matrix_lr, momentum: 0.95, ns_steps: 5, beta2: 0.95, weight_decay: weight_decay)
    end
    MuonAdamW.new(named.map(&.[1]), groups)
  end

  # Returns the logits, `[b * t, vocab_size]`, for token ids *idx*, `[b, t]`; or,
  # given *targets*, `[b, t]`, their cross-entropy, ignoring targets of -1.
  def forward(idx : Tensor, targets : Tensor? = nil, reduction : Reduction = :mean) : Tensor
    b, t = idx.shape[0], idx.shape[1]
    ids = idx.reshape([b * t])
    cos, sin = @rope.angles((0...t).to_a)
    cos_sin = {cos.reshape([1, t, 1, cos.shape[-1]]), (sin * -1.0).reshape([1, t, 1, sin.shape[-1]])}

    x = norm(@transformer.wte.call(ids))
    x0 = x
    @transformer.h.each_with_index do |block, i|
      x = x * @resid_lambdas.value.slice(0, i, i + 1) + x0 * @x0_lambdas.value.slice(0, i, i + 1)
      ve = @value_embeds[i.to_s]?.try(&.call(ids))
      x = block.call(x, ve, cos_sin, mask(t, window_sizes[i]), b, t)
    end
    x = norm(x)

    logits = @lm_head.call(x)
    logits = (logits / SOFTCAP).tanh * SOFTCAP
    return logits unless targets
    logits.cross_entropy(targets.reshape([b * t]), reduction: reduction, ignore_index: -1)
  end

  private def mask(t : Int32, window : Int32) : Tensor
    @masks[{t, window}] ||= window_mask(t, window, @device)
  end

  # Returns the parameters whose names start with *prefix*, in model order. Names
  # come from this model, not a child's `params`, which would rename them.
  private def params_named(prefix : String) : Array(Parameter)
    named_params.select { |name, _| name.starts_with?(prefix) }.values
  end

  private def count(parameters : Array(Parameter)) : Int64
    parameters.sum(0i64, &.shape.product(1i64))
  end
end

# Replaces *param*'s value with one drawn from *init*, on the parameter's device.
def fill(param : Parameter, init : Init, rng : Random) : Nil
  param.value = init.tensor(param.shape, rng, param.value.device)
end

# Returns the config for a model of *depth* layers: `model_dim` is
# `depth * ASPECT_RATIO` rounded up to a multiple of `HEAD_DIM`.
def build_model_config(depth : Int32) : GPTConfig
  base_dim = depth * ASPECT_RATIO
  model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
  num_heads = model_dim // HEAD_DIM
  GPTConfig.new(
    sequence_len: Autoresearch::MAX_SEQ_LEN, vocab_size: Autoresearch::VOCAB_SIZE,
    n_layer: depth, n_head: num_heads, n_kv_head: num_heads, n_embd: model_dim,
    window_pattern: WINDOW_PATTERN,
  )
end

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW)
# ---------------------------------------------------------------------------

POLAR_EXPRESS_COEFFS = [
  {8.156554524902461, -22.48329292557795, 15.878769915207462},
  {4.042929935166739, -2.808917465908714, 0.5000178451051316},
  {3.8916678022926607, -2.772484153217685, 0.5060648178503393},
  {3.285753657755655, -2.3681294933425376, 0.46449024233003106},
  {2.3465413258596377, -1.7097828382687081, 0.42323551169305323},
]

# Parameters stepped by AdamW, by index into the optimizer's parameters, with
# their settings. The schedule sets `lr` from `initial_lr` each step.
class AdamWGroup
  getter params : Array(Int32)
  property lr : Float64
  getter initial_lr : Float64
  getter betas : {Float64, Float64}
  getter eps : Float64
  property weight_decay : Float64

  def initialize(@params, @lr, @betas, @eps, @weight_decay)
    @initial_lr = @lr
  end
end

# Matrices of one shape stepped together by Muon, by index into the optimizer's
# parameters, with their settings. The schedule sets `lr`, `momentum` and
# `weight_decay` each step.
class MuonGroup
  getter params : Array(Int32)
  property lr : Float64
  getter initial_lr : Float64
  property momentum : Float64
  getter ns_steps : Int32
  getter beta2 : Float64
  property weight_decay : Float64

  def initialize(@params, @lr, *, @momentum, @ns_steps, @beta2, @weight_decay)
    @initial_lr = @lr
  end
end

alias ParamGroup = AdamWGroup | MuonGroup

# train.py's combined optimizer: AdamW for each parameter of an `AdamWGroup`,
# and Muon for each `MuonGroup`, whose matrices are stacked and orthogonalised
# together. Muon works on `[out, in]` matrices, as train.py does: it transposes
# tensomite's `[in, out]` weights and transposes the result back.
#
# State is kept per parameter index (`exp_avg`, `exp_avg_sq`) and per Muon
# group, under its first parameter's index (`momentum_buffer`,
# `second_momentum_buffer`), so `#state` and `#restore` resume a run exactly.
class MuonAdamW < Optimizer
  getter param_groups : Array(ParamGroup)
  @exp_avg = {} of Int32 => Tensor
  @exp_avg_sq = {} of Int32 => Tensor
  @momentum_buffer = {} of Int32 => Tensor
  @second_momentum_buffer = {} of Int32 => Tensor

  # Raises `ArgumentError` unless every one of *params* is in exactly one of *param_groups*.
  def initialize(params : Array(Parameter), @param_groups : Array(ParamGroup))
    @slots = params
    covered = param_groups.flat_map(&.params).sort!
    unless covered == (0...params.size).to_a
      raise ArgumentError.new("param groups must hold each of #{params.size} params once, not #{covered}")
    end
  end

  protected def tables : Hash(String, Hash(Int32, Tensor))
    {"exp_avg" => @exp_avg, "exp_avg_sq" => @exp_avg_sq,
     "momentum_buffer" => @momentum_buffer, "second_momentum_buffer" => @second_momentum_buffer}
  end

  protected def update(index : Int32, param : Tensor, grad : Tensor) : Tensor
    raise Error.new("MuonAdamW steps its groups in #updates")
  end

  protected def updates(params : Array(Tensor), grads : Array(Tensor)) : Array(Tensor)
    result = Array(Tensor?).new(params.size, nil)
    param_groups.each do |group|
      case group
      in AdamWGroup
        group.params.each { |i| result[i] = step_adamw(group, i, params[i], grads[i]) }
      in MuonGroup
        step_muon(group, params, grads).each_with_index { |updated, member| result[group.params[member]] = updated }
      end
    end
    result.map_with_index { |updated, i| updated || raise Error.new("no group updated param #{i}") }
  end

  # Returns param *i* after one AdamW step:
  # 1. decay the param by `lr × weight_decay`;
  # 2. move the first and second moments towards the gradient and its square;
  # 3. step by the bias-corrected first moment over the root of the second.
  private def step_adamw(group : AdamWGroup, i : Int32, p : Tensor, grad : Tensor) : Tensor
    beta1, beta2 = group.betas
    p = p * (1 - group.lr * group.weight_decay)
    exp_avg = keep(@exp_avg, i, lerp(@exp_avg[i]?, grad, 1 - beta1))
    exp_avg_sq = keep(@exp_avg_sq, i, lerp(@exp_avg_sq[i]?, grad * grad, 1 - beta2))
    bias1 = 1 - beta1 ** steps
    bias2 = 1 - beta2 ** steps
    denom = (exp_avg_sq / bias2).sqrt + group.eps
    p - (exp_avg / denom) * (group.lr / bias1)
  end

  # Returns the group's params after one Muon step, in group order:
  # 1. Nesterov momentum on the stacked `[out, in]` gradients;
  # 2. polar-express orthogonalisation, `ns_steps` Newton–Schulz iterations;
  # 3. NorMuon: scale by a running second moment along the smaller axis,
  #    keeping the update's overall norm;
  # 4. cautious weight decay, only where the update and the param agree in sign.
  private def step_muon(group : MuonGroup, params : Array(Tensor), grads : Array(Tensor)) : Array(Tensor)
    first = group.params.first
    stacked_grads = Tensor.stack(group.params.map { |i| grads[i].transpose })
    stacked_params = Tensor.stack(group.params.map { |i| params[i].transpose })
    rows, cols = stacked_grads.shape[-2], stacked_grads.shape[-1]
    red_dim = rows >= cols ? -1 : -2

    momentum = group.momentum
    momentum_buffer = keep(@momentum_buffer, first, lerp(@momentum_buffer[first]?, stacked_grads, 1 - momentum))
    g = stacked_grads + (momentum_buffer - stacked_grads) * momentum

    x = g / (frobenius(g) * 1.02 + 1e-6)
    POLAR_EXPRESS_COEFFS.first(group.ns_steps).each do |(a, b, c)|
      if rows > cols
        gram = x.transpose.matmul(x)
        x = x * a + x.matmul(gram * b + gram.matmul(gram) * c)
      else
        gram = x.matmul(x.transpose)
        x = x * a + (gram * b + gram.matmul(gram) * c).matmul(x)
      end
    end
    g = x

    floor = Tensor.scalar(1e-10, DType::F32, g.device)
    v_mean = g.square.mean(red_dim, true)
    red_dim_size = g.shape[red_dim]
    v_norm = (v_mean.sum(-1, true).sum(-2, true) * red_dim_size).sqrt
    second = keep(@second_momentum_buffer, first, lerp(@second_momentum_buffer[first]?, v_mean, 1 - group.beta2))
    step_size = second.maximum(floor).rsqrt
    v_norm_new = (v_mean * red_dim_size * step_size.square).sum(-1, true).sum(-2, true).sqrt
    g = g * (step_size * (v_norm / v_norm_new.maximum(floor)))

    lr = group.lr * Math.max(1.0, rows / cols) ** 0.5
    mask = (g * stacked_params).greater_equal(Tensor.scalar(0.0, DType::F32, g.device))
    updated = stacked_params - (g * lr + stacked_params * mask * (lr * group.weight_decay))
    updated.unbind(0).map(&.transpose.contiguous)
  end

  # Returns `previous + (target - previous) × weight`, from zero when there is no *previous*.
  private def lerp(previous : Tensor?, target : Tensor, weight : Float64) : Tensor
    previous ? previous + (target - previous) * weight : target * weight
  end

  # Returns the Frobenius norm of each matrix in the stack *x*, `[n, 1, 1]`.
  private def frobenius(x : Tensor) : Tensor
    x.square.sum(-1, true).sum(-2, true).sqrt
  end
end

# Returns the learning-rate multiplier at *progress* through the time budget:
# a linear warm-up, a plateau at 1, then a linear warm-down to `FINAL_LR_FRAC`.
def get_lr_multiplier(progress : Float64) : Float64
  if progress < WARMUP_RATIO
    WARMUP_RATIO > 0 ? progress / WARMUP_RATIO : 1.0
  elsif progress < 1.0 - WARMDOWN_RATIO
    1.0
  else
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
  end
end

# Returns Muon's momentum at *step*: from 0.85 to 0.95 over the first 300 steps.
def get_muon_momentum(step : Int32) : Float64
  frac = Math.min(step / 300, 1)
  (1 - frac) * 0.85 + frac * 0.95
end

# Returns Muon's weight decay at *progress*: `WEIGHT_DECAY`, falling linearly to zero.
def get_weight_decay(progress : Float64) : Float64
  WEIGHT_DECAY * (1 - progress)
end

# Sets every group's schedule-driven settings for the step at *progress* and *step*, as train.py's loop does.
def apply_schedules(optimizer : MuonAdamW, progress : Float64, step : Int32) : Nil
  lrm = get_lr_multiplier(progress)
  muon_momentum = get_muon_momentum(step)
  muon_weight_decay = get_weight_decay(progress)
  optimizer.param_groups.each do |group|
    group.lr = group.initial_lr * lrm
    if group.is_a?(MuonGroup)
      group.momentum = muon_momentum
      group.weight_decay = muon_weight_decay
    end
  end
end

# ---------------------------------------------------------------------------
# Setup and training
# ---------------------------------------------------------------------------

def main : Nil
  device = BLAS.default.as(Device)
  config = build_model_config(DEPTH)
  puts "Vocab size: #{config.vocab_size}"
  puts "Model config: #{config}"
  model = GPT.new(config, device)
  model.init_weights(Random.new(42))
  puts "Parameter counts:"
  model.num_scaling_params.each { |key, value| puts "  #{key.ljust(24)}: #{value.format}" }
  puts "Estimated FLOPs per token: #{"%e" % model.estimate_flops.to_f}"
end

{% unless flag?(:autoresearch_spec) %}
  main
{% end %}
