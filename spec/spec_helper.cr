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
