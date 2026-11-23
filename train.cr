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
    nparams_exclude = count(@transformer.wte.params + @value_embeds.values.flat_map(&.params) + [@resid_lambdas, @x0_lambdas])
    h = config.n_head.to_i64
    q = (config.n_embd // config.n_head).to_i64
    t = config.sequence_len
    attn_flops = window_sizes.sum(0i64) { |window| 12i64 * h * q * (window < 0 ? t : Math.min(window, t)) }
    6i64 * (nparams - nparams_exclude) + attn_flops
  end

  # Returns the parameter counts by group, as train.py's `num_scaling_params`.
  def num_scaling_params : Hash(String, Int64)
    counts = {
      "wte"                  => count(@transformer.wte.params),
      "value_embeds"         => count(@value_embeds.values.flat_map(&.params)),
      "lm_head"              => count(@lm_head.params),
      "transformer_matrices" => count(@transformer.h.flat_map(&.params)),
      "scalars"              => count([@resid_lambdas, @x0_lambdas]),
    }
    counts["total"] = counts.values.sum
    counts
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
