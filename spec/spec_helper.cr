require "spec"
require "json"

{% unless flag?(:autoresearch_spec) %}
  {% raise "specs need -Dautoresearch_spec, so that requiring train.cr does not run it: use `make spec`" %}
{% end %}

require "../train"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "tensomite/metal"
{% end %}
{% if flag?(:cuda) %}
  require "tensomite/cuda"
{% end %}

# The host device for specs that run a model. `CPU.default` is the slow oracle, for op-level checks only.
DEVICE = BLAS.default.as(Device)

# GPUs present on this machine: Metal on Apple Silicon, CUDA in a build with `-Dcuda` (`make spec CUDA=1`).
GPUS = {% if flag?(:darwin) && flag?(:aarch64) %}
         Metal.available? ? [Metal.default] of Device : [] of Device
       {% elsif flag?(:cuda) %}
         CUDA.available? ? [CUDA.default] of Device : [] of Device
       {% else %}
         [] of Device
       {% end %}

# Every device the parity specs run the model on: `DEVICE`, then each of `GPUS`.
MODEL_DEVICES = [DEVICE] + GPUS

# Builds a tensor of *data* with *shape* on `DEVICE`.
def tensor(shape : Array(Int32), data : Array(T)) : Tensor forall T
  Tensor.from(data, shape, DEVICE)
end

# Checks *actual* against *expected* element by element, within an absolute *tolerance*.
def expect_close(actual : Array(Float32), expected : Array(Float32), tolerance : Float64) : Nil
  actual.size.should eq expected.size
  actual.zip(expected) { |a, e| a.should be_close(e, tolerance) }
end

# Checks that no element of *actual* is further from *expected* than *relative*
# times the largest magnitude in *expected*, and names the worst element if one is.
def expect_close_to_scale(actual : Array(Float32), expected : Array(Float32), relative : Float64, label : String = "") : Nil
  actual.size.should eq expected.size
  scale = expected.max_of(&.abs).to_f64
  worst, at = actual.zip(expected).each_with_index.max_of { |(a, e), i| {(a - e).abs.to_f64, i} }
  unless worst <= relative * scale
    fail "#{label}: element #{at} is #{actual[at]}, expected #{expected[at]}; |difference| #{worst} > #{relative} × #{scale}"
  end
end

# Fixtures from scripts/fixtures/make_fixtures.py: train.py's model at the
# pinned commit, run by PyTorch on the CPU in float32 (see spec/fixtures/README.md).
MODEL_FIXTURES = "#{__DIR__}/fixtures/model"

# Returns *tensor*, a parameter named *name* or its gradient, in the other
# layout: PyTorch's `[out, in]` for a Linear weight in tensomite's `[in, out]`,
# and back. Other tensors are returned as they are.
def other_layout(name : String, tensor : Tensor) : Tensor
  name.matches?(TRANSPOSED) ? tensor.transpose.contiguous : tensor
end

# Linear weights, which PyTorch stores as [out, in] and tensomite as [in, out].
TRANSPOSED = /(c_q|c_k|c_v|c_proj|c_fc|ve_gate|lm_head)\.weight$/

# Returns the model fixtures' `expected.json`.
def model_meta : JSON::Any
  JSON.parse(File.read("#{MODEL_FIXTURES}/expected.json"))
end

# Returns the config of the tiny fixture model.
def tiny_config : GPTConfig
  t = model_meta["tiny"]
  GPTConfig.new(
    sequence_len: t["sequence_len"].as_i, vocab_size: t["vocab_size"].as_i, n_layer: t["n_layer"].as_i,
    n_head: t["n_head"].as_i, n_kv_head: t["n_kv_head"].as_i, n_embd: t["n_embd"].as_i,
    window_pattern: t["window_pattern"].as_s,
  )
end

# Returns the tiny fixture model on *device*, with the fixture's weights.
def fixture_model(device : Device) : GPT
  GPT.new(tiny_config, device).load("#{MODEL_FIXTURES}/weights.safetensors", transposed: TRANSPOSED)
end
