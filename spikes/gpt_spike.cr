require "tensomite"
require "tensomite/blas"

include Tensomite

DEVICE = Tensomite::BLAS.default.as(Tensomite::Device)
RMS_EPS = Float32::EPSILON.to_f64
SOFTCAP = 15.0

def norm(x : Tensor) : Tensor
  x.rmsnorm(RMS_EPS)
end

# Causal mask with a sliding window: position i sees j when j <= i and i - j < window.
def window_mask(t : Int32, window : Int32) : Tensor
  data = Array(Float32).new(t * t) do |n|
    i, j = n // t, n % t
    j <= i && i - j < window ? 0f32 : -Float32::INFINITY
  end
  Tensor.from(data, [t, t], DEVICE)
end

class Attn < Tensomite::Module
  @c_q : Linear
  @c_k : Linear
  @c_v : Linear
  @c_proj : Linear
  @ve_gate : Linear?

  def initialize(@width : Int32, @heads : Int32, ve : Bool, build : Builder)
    s = Math.sqrt(3.0) / Math.sqrt(width.to_f)
    uniform = Init.uniform(s)
    @c_q = build.linear(width, width, bias: false, init: uniform)
    @c_k = build.linear(width, width, bias: false, init: uniform)
    @c_v = build.linear(width, width, bias: false, init: uniform)
    @c_proj = build.linear(width, width, bias: false, init: Init.constant(0))
    @ve_gate = ve ? build.linear(32, heads, bias: false, init: Init.constant(0)) : nil
  end

  def forward(x : Tensor, ve : Tensor?, cos : Tensor, sin : Tensor, mask : Tensor, b : Int32, t : Int32) : Tensor
    h, d = @heads, @width // @heads
    q = @c_q.call(x).reshape([b, t, h, d])
    k = @c_k.call(x).reshape([b, t, h, d])
    v = @c_v.call(x).reshape([b, t, h, d])
    if (gate = @ve_gate) && ve
      g = gate.call(x.slice(1, 0, 32)).sigmoid * 2.0
      v = v + g.reshape([b, t, h, 1]) * ve.reshape([b, t, h, d])
    end
    q = norm(q.rope(cos, sin)).permute([0, 2, 1, 3])
    k = norm(k.rope(cos, sin)).permute([0, 2, 1, 3])
    v = v.permute([0, 2, 1, 3])
    probs = q.matmul(k.transpose).softmax(scale: 1.0 / Math.sqrt(d.to_f), mask: mask)
    y = probs.matmul(v).permute([0, 2, 1, 3]).contiguous.reshape([b * t, @width])
    @c_proj.call(y)
  end
end

class MLP < Tensomite::Module
  @c_fc : Linear
  @c_proj : Linear

  def initialize(width : Int32, build : Builder)
    s = Math.sqrt(3.0) / Math.sqrt(width.to_f)
    @c_fc = build.linear(width, 4 * width, bias: false, init: Init.uniform(s))
    @c_proj = build.linear(4 * width, width, bias: false, init: Init.constant(0))
  end

  def forward(x : Tensor) : Tensor
    @c_proj.call(@c_fc.call(x).relu.square)
  end
end

class Block < Tensomite::Module
  @attn : Attn
  @mlp : MLP

  def initialize(width : Int32, heads : Int32, ve : Bool, build : Builder)
    @attn = Attn.new(width, heads, ve, build)
    @mlp = MLP.new(width, build)
  end

  def forward(x : Tensor, ve : Tensor?, cos : Tensor, sin : Tensor, mask : Tensor, b : Int32, t : Int32) : Tensor
    x = x + @attn.call(norm(x), ve, cos, sin, mask, b, t)
    x + @mlp.call(norm(x))
  end
end

class GPT < Tensomite::Module
  getter layers : Int32
  getter width : Int32
  @wte : Embedding
  @h : Array(Block)
  @value_embeds : Array(Embedding)
  @lm_head : Linear
  @resid_lambdas : Parameter
  @x0_lambdas : Parameter

  def self.ve?(layer : Int32, layers : Int32) : Bool
    layer % 2 == (layers - 1) % 2
  end

  def initialize(@layers : Int32, @width : Int32, @heads : Int32, @vocab : Int32, @context : Int32, @pattern : String, build : Builder)
    s = Math.sqrt(3.0) / Math.sqrt(width.to_f)
    @wte = build.embedding(@vocab, width, init: Init.normal(1.0))
    @h = Array(Block).new(layers) { |i| Block.new(width, heads, GPT.ve?(i, layers), build) }
    @value_embeds = (0...layers).select { |i| GPT.ve?(i, layers) }.map { build.embedding(@vocab, width, init: Init.uniform(s)) }
    @lm_head = build.linear(width, @vocab, bias: false, init: Init.normal(0.001))
    @resid_lambdas = build.parameter([layers], init: Init.constant(1.0), decay: false)
    @x0_lambdas = build.parameter([layers], init: Init.constant(0.1), decay: false)
    @rope = RotaryEmbedding.new(width // heads, context: context, device: DEVICE)
    @masks = {} of Int32 => Tensor
  end

  def window(layer : Int32) : Int32
    return @context if layer == @layers - 1
    @pattern[layer % @pattern.size].upcase == 'S' ? @context // 2 : @context
  end

  def forward(ids : Tensor, b : Int32, t : Int32) : Tensor
    cos, sin = @rope.angles((0...t).to_a)
    cos = cos.reshape([1, t, 1, cos.shape.last])
    sin = (sin * -1.0).reshape([1, t, 1, sin.shape.last])
    x = norm(@wte.call(ids))
    x0 = x
    ve_index = 0
    @h.each_with_index do |block, i|
      x = x * @resid_lambdas.value.slice(0, i, i + 1) + x0 * @x0_lambdas.value.slice(0, i, i + 1)
      ve = nil
      if GPT.ve?(i, @layers)
        ve = @value_embeds[ve_index].call(ids)
        ve_index += 1
      end
      mask = (@masks[window(i)] ||= window_mask(t, window(i)))
      x = block.call(x, ve, cos, sin, mask, b, t)
    end
    logits = @lm_head.call(norm(x))
    (logits / SOFTCAP).tanh * SOFTCAP
  end
end

# Muon for the blocks' matrices, stacked by shape, and AdamW for everything else.
class MuonAdamW < Optimizer
  COEFFS = [
    {8.156554524902461, -22.48329292557795, 15.878769915207462},
    {4.042929935166739, -2.808917465908714, 0.5000178451051316},
    {3.8916678022926607, -2.772484153217685, 0.5060648178503393},
    {3.285753657755655, -2.3681294933425376, 0.46449024233003106},
    {2.3465413258596377, -1.7097828382687081, 0.42323551169305323},
  ]

  record Adam, lr : Float64, beta1 : Float64, beta2 : Float64, eps : Float64

  property scale = 1.0
  property momentum = 0.95
  property weight_decay = 0.0
  property muon_lr : Float64
  property muon_beta2 = 0.95

  @m = {} of Int32 => Tensor
  @v = {} of Int32 => Tensor
  @momentum_buffer = {} of Int32 => Tensor
  @second_buffer = {} of Int32 => Tensor

  # *settings* holds one Adam setting per param, or nil for a matrix that Muon updates.
  def initialize(params : Array(Parameter), @settings : Array(Adam?), @muon_lr : Float64)
    @slots = params
  end

  protected def tables : Hash(String, Hash(Int32, Tensor))
    {"m" => @m, "v" => @v, "momentum" => @momentum_buffer, "second" => @second_buffer}
  end

  protected def update(index : Int32, param : Tensor, grad : Tensor) : Tensor
    raise "unused"
  end

  protected def updates(params : Array(Tensor), grads : Array(Tensor)) : Array(Tensor)
    result = Array(Tensor?).new(params.size, nil)
    groups = {} of Array(Int32) => Array(Int32)
    params.each_index do |i|
      if setting = @settings[i]
        result[i] = adam(i, params[i], grads[i], setting)
      else
        (groups[params[i].shape] ||= [] of Int32) << i
      end
    end
    groups.each_value { |indices| muon(indices, params, grads, result) }
    result.map(&.not_nil!)
  end

  private def adam(i : Int32, p : Tensor, g : Tensor, s : Adam) : Tensor
    lr = s.lr * scale
    m = (prev = @m[i]?) ? prev + (g - prev) * (1 - s.beta1) : g * (1 - s.beta1)
    v = (prev = @v[i]?) ? prev + (g * g - prev) * (1 - s.beta2) : g * g * (1 - s.beta2)
    keep(@m, i, m)
    keep(@v, i, v)
    denom = (v / (1 - s.beta2 ** steps)).sqrt + s.eps
    p - (m / denom) * (lr / (1 - s.beta1 ** steps))
  end

  private def muon(indices : Array(Int32), params : Array(Tensor), grads : Array(Tensor), result : Array(Tensor?)) : Nil
    first = indices.first
    # tensomite stores weights [in, out]; the recipe is written for [out, in].
    g = Tensor.stack(indices.map { |i| grads[i].transpose })
    p = Tensor.stack(indices.map { |i| params[i].transpose })
    rows, cols = g.shape[-2], g.shape[-1]

    buffer = (prev = @momentum_buffer[first]?) ? prev + (g - prev) * (1 - momentum) : g * (1 - momentum)
    keep(@momentum_buffer, first, buffer)
    g = g + (buffer - g) * momentum

    norms = (g * g).sum(-1, true).sum(-2, true).sqrt
    x = g / (norms * 1.02 + 1e-6)
    COEFFS.each do |(a, b, c)|
      if rows > cols
        gram = x.transpose.matmul(x)
        x = x * a + x.matmul(gram * b + gram.matmul(gram) * c)
      else
        gram = x.matmul(x.transpose)
        x = x * a + (gram * b + gram.matmul(gram) * c).matmul(x)
      end
    end

    red = rows >= cols ? -1 : -2
    red_size = x.shape[red]
    v_mean = (x * x).mean(red, true)
    v_norm = (v_mean.sum(-1, true).sum(-2, true) * red_size).sqrt
    second = (prev = @second_buffer[first]?) ? prev + (v_mean - prev) * (1 - muon_beta2) : v_mean * (1 - muon_beta2)
    keep(@second_buffer, first, second)
    floor = Tensor.scalar(1e-10, DType::F32, second.device)
    step_size = second.maximum(floor).rsqrt
    v_norm_new = (v_mean * red_size * step_size.square).sum(-1, true).sum(-2, true).sqrt
    x = x * (step_size * (v_norm / v_norm_new.maximum(floor)))

    lr = muon_lr * scale * Math.max(1.0, rows.to_f / cols) ** 0.5
    zero = Tensor.scalar(0.0, DType::F32, x.device)
    mask = (x * p).greater_equal(zero)
    updated = p - (x * lr + p * mask * (lr * weight_decay))
    updated.unbind(0).each_with_index { |slice, n| result[indices[n]] = slice.transpose.contiguous }
  end
end

# --- A learnable stream: each token is a fixed function of the one before it.
def batch(rng : Random, b : Int32, t : Int32, vocab : Int32) : {Tensor, Tensor}
  inputs = Array(Int32).new(b * t)
  targets = Array(Int32).new(b * t)
  b.times do
    x = rng.rand(vocab)
    t.times do
      y = (x * 5 + 3) % vocab
      inputs << x
      targets << y
      x = y
    end
  end
  {Tensor.from(inputs, [b * t], DEVICE), Tensor.from(targets, [b * t], DEVICE)}
end

layers, width, heads, vocab, t, b = 4, 128, 4, 256, 64, 8
build = Builder.new(rng: Random.new(1), device: DEVICE)
model = GPT.new(layers, width, heads, vocab, t, "SL", build)
named = model.named_params.to_a
puts "params: #{named.sum { |(_, p)| p.shape.product }}  tensors: #{named.size}"
named.first(6).each { |(n, p)| puts "  #{n} #{p.shape}" }

lr_scale = (width / 768.0) ** -0.5
adam_betas = {0.8, 0.95}
settings = named.map do |(name, p)|
  if name.starts_with?("h.")
    nil
  elsif name.starts_with?("lm_head")
    MuonAdamW::Adam.new(0.004 * lr_scale, adam_betas[0], adam_betas[1], 1e-10)
  elsif name.starts_with?("wte") || name.starts_with?("value_embeds")
    MuonAdamW::Adam.new(0.6 * lr_scale, adam_betas[0], adam_betas[1], 1e-10)
  elsif name.starts_with?("resid")
    MuonAdamW::Adam.new(0.5 * 0.01, adam_betas[0], adam_betas[1], 1e-10)
  else
    MuonAdamW::Adam.new(0.5, 0.96, 0.95, 1e-10)
  end
end
optimizer = MuonAdamW.new(named.map(&.[1]), settings, 0.04)

rng = Random.new(7)
steps = (ARGV[0]? || "200").to_i
started = Time.instant
steps.times do |step|
  progress = step / steps.to_f
  optimizer.scale = progress < 0.5 ? 1.0 : (1.0 - progress) / 0.5
  optimizer.momentum = 0.85 + 0.1 * Math.min(step / 300.0, 1.0)
  optimizer.weight_decay = 0.2 * (1 - progress)
  ids, targets = batch(rng, b, t, vocab)
  if ARGV[1]? == "accum"
    values = named.map(&.[1].value)
    micro = 4
    parts = b // micro
    sums = nil
    last = 0f32
    parts.times do |k|
      mi = ids.slice(0, k * micro * t, (k + 1) * micro * t)
      mt = targets.slice(0, k * micro * t, (k + 1) * micro * t)
      micro_loss = model.call(mi, micro, t).cross_entropy(mt)
      grads = Tensomite.grad(micro_loss, values)
      Tensor.realize(grads + [micro_loss], release: true)
      grads = grads.map(&.detach)
      last += micro_loss.to_a(Float32).first / parts
      sums = sums ? sums.zip(grads).map { |(a, g)| a + g } : grads
    end
    mean = sums.not_nil!.map { |g| g * (1.0 / parts) }
    updated = optimizer.step(values, mean)
    named.each_with_index { |(_, param), i| param.value = updated[i] }
    loss = last
  else
    loss = optimizer.step(model.call(ids, b, t).cross_entropy(targets))
  end
  puts "step #{step}: loss #{loss.round(4)}" if step % 20 == 0 || step == steps - 1
  abort "NaN" if loss.nan?
end
puts "#{(Time.instant - started).total_seconds.round(1)} s"
