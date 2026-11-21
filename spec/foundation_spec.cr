require "./spec_helper"

describe "M0 exit: foundation" do
  it "uses tensomite 0.8" do
    Tensomite::VERSION.should start_with "0.8."
  end

  it "multiplies matrices on the BLAS device" do
    a = tensor([2, 2], [1, 2, 3, 4].map(&.to_f32))
    b = tensor([2, 2], [5, 6, 7, 8].map(&.to_f32))
    # [[1·5 + 2·7, 1·6 + 2·8], [3·5 + 4·7, 3·6 + 4·8]] = [[19, 22], [43, 50]]
    a.matmul(b).to_a(Float32).should eq [19, 22, 43, 50].map(&.to_f32)
  end

  it "compares within a tolerance" do
    expect_close([1.0f32, 2.0f32], [1.0f32, 2.0005f32], tolerance: 1e-3)
  end
end
