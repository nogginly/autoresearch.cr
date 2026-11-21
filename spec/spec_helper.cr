require "spec"
require "tensomite"
require "tensomite/blas"

include Tensomite

# The device for every spec that runs a model. `CPU.default` is the slow oracle, for op-level checks only.
DEVICE = BLAS.default.as(Device)

# Builds a tensor of *data* with *shape* on `DEVICE`.
def tensor(shape : Array(Int32), data : Array(T)) : Tensor forall T
  Tensor.from(data, shape, DEVICE)
end

# Checks *actual* against *expected* element by element, within an absolute *tolerance*.
def expect_close(actual : Array(Float32), expected : Array(Float32), tolerance : Float64) : Nil
  actual.size.should eq expected.size
  actual.zip(expected) { |a, e| a.should be_close(e, tolerance) }
end
