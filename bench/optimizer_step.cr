# Measures MuonAdamW's step against a training step of the depth-8 model, on one device.
#
# Usage: crystal run --release -Dautoresearch_spec bench/optimizer_step.cr -- [blas|metal|cuda] [batch]
# (`make bench DEVICE=… BATCH=…`). CUDA needs a `-Dcuda` build (`CUDA=1`).
#
# A training step is train.py's 2^19 tokens. The benchmark times forward and
# backward on one micro-batch of *batch* sequences of `MAX_SEQ_LEN` tokens and
# scales it to 2^19 tokens, since a full step is too slow to time on the host.
require "../train"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "tensomite/metal"
{% end %}
{% if flag?(:cuda) %}
  require "tensomite/cuda"
{% end %}

TOTAL_TOKENS = 2 ** 19
REPEATS      = 3

def pick_device(name : String) : Device
  return BLAS.default if name == "blas"
  {% if flag?(:darwin) && flag?(:aarch64) %}
    return Metal.default if name == "metal"
  {% end %}
  {% if flag?(:cuda) %}
    return CUDA.default if name == "cuda"
  {% end %}
  abort "unknown or unavailable device #{name}: use blas, metal (Apple Silicon) or cuda (a -Dcuda build)"
end

# Returns the median of *REPEATS* timings of the block, after one untimed warm-up.
def median_seconds(& : -> Nil) : Float64
  yield
  GC.collect
  times = Array.new(REPEATS) do
    started = Time.instant
    yield
    elapsed = (Time.instant - started).total_seconds
    GC.collect
    elapsed
  end
  times.sort![REPEATS // 2]
end

device = pick_device(ARGV[0]? || "blas")
batch = (ARGV[1]? || "1").to_i
config = build_model_config(DEPTH)
t = config.sequence_len
rng = Random.new(1)

model = GPT.new(config, device)
model.init_weights(rng)
optimizer = model.setup_optimizer(unembedding_lr: UNEMBEDDING_LR, embedding_lr: EMBEDDING_LR, scalar_lr: SCALAR_LR,
  adam_betas: ADAM_BETAS, matrix_lr: MATRIX_LR, weight_decay: WEIGHT_DECAY)
values = model.params.map(&.value)
idx = Tensor.from(Array(Int32).new(batch * t) { rng.rand(config.vocab_size) }, [batch, t], device)
targets = Tensor.from(Array(Int32).new(batch * t) { rng.rand(config.vocab_size) }, [batch, t], device)

grads = [] of Tensor
forward_backward = median_seconds do
  loss = model.call(idx, targets)
  grads = Tensomite.grad(loss, values)
  Tensor.realize(grads + [loss], release: true)
  grads = grads.map(&.detach)
end

step = 0
optimizer_step = median_seconds do
  apply_schedules(optimizer, 0.0, step)
  step += 1
  values = optimizer.step(values, grads)
end

training_step = forward_backward / (batch * t) * TOTAL_TOKENS
puts "device:                 #{device.class.name}"
puts "micro-batch:            #{batch} × #{t} tokens"
puts "forward and backward:   #{forward_backward.round(3)} s"
puts "training step (2^19):   #{training_step.round(1)} s, scaled from the micro-batch"
puts "optimizer step:         #{optimizer_step.round(3)} s"
puts "optimizer / training:   #{(100 * optimizer_step / training_step).round(3)} %"
