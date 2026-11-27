require "tensomite"
require "tensomite/blas"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "tensomite/metal"
{% end %}
{% if flag?(:cuda) %}
  require "tensomite/cuda"
{% end %}

# The tensomite devices a program can pick by name. `cpu` is the naive oracle,
# too slow to train on.
module Autoresearch
  # Returns the names of the devices this build supports: `cpu`, `blas`, then
  # `metal` on Apple Silicon and `cuda` in a `-Dcuda` build.
  def self.device_names : Array(String)
    names = ["cpu", "blas"]
    {% if flag?(:darwin) && flag?(:aarch64) %}
      names << "metal"
    {% end %}
    {% if flag?(:cuda) %}
      names << "cuda"
    {% end %}
    names
  end

  # Returns the default device named *name*, one of `device_names`. Raises
  # `ArgumentError` for any other name.
  def self.device(name : String) : Tensomite::Device
    return Tensomite::CPU.default if name == "cpu"
    return Tensomite::BLAS.default if name == "blas"
    {% if flag?(:darwin) && flag?(:aarch64) %}
      return Tensomite::Metal.default if name == "metal"
    {% end %}
    {% if flag?(:cuda) %}
      return Tensomite::CUDA.default if name == "cuda"
    {% end %}
    raise ArgumentError.new("unknown or unavailable device #{name}: use #{device_names.join(", ")}")
  end
end
