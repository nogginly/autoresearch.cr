# Shared by the benchmarks: the devices this build has, and a model to run on them.
require "../train"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "tensomite/metal"
{% end %}
{% if flag?(:cuda) %}
  require "tensomite/cuda"
{% end %}

# Returns the device named *name*, or aborts naming those this build has.
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

# Returns `build_model_config(depth)` with *sequence_len* in place of `MAX_SEQ_LEN`.
def bench_config(depth : Int32, sequence_len : Int32) : GPTConfig
  build_model_config(depth).copy_with(sequence_len: sequence_len)
end

# Returns a model of *config* on *device* with train.py's initial weights, its
# optimizer with train.py's hyperparameters, and a random batch of *batch*
# sequences: ids and targets.
def bench_setup(config : GPTConfig, device : Device, batch : Int32) : {GPT, MuonAdamW, Tensor, Tensor}
  rng = Random.new(1)
  model = GPT.new(config, device)
  model.init_weights(rng)
  optimizer = model.setup_optimizer(unembedding_lr: UNEMBEDDING_LR, embedding_lr: EMBEDDING_LR, scalar_lr: SCALAR_LR,
    adam_betas: ADAM_BETAS, matrix_lr: MATRIX_LR, weight_decay: WEIGHT_DECAY)
  t = config.sequence_len
  idx = Tensor.from(Array(Int32).new(batch * t) { rng.rand(config.vocab_size) }, [batch, t], device)
  targets = Tensor.from(Array(Int32).new(batch * t) { rng.rand(config.vocab_size) }, [batch, t], device)
  {model, optimizer, idx, targets}
end
