require "./spec_helper"
require "../src/autoresearch/device"

describe "Autoresearch.device" do
  it "gives the default BLAS and CPU devices" do
    Autoresearch.device("blas").should be(BLAS.default)
    Autoresearch.device("cpu").should be(CPU.default)
  end

  it "rejects an unknown name, listing the devices this build has" do
    error = expect_raises(ArgumentError, /unknown or unavailable device nope/) { Autoresearch.device("nope") }
    message = error.message.to_s
    Autoresearch.device_names.each { |name| message.should contain(name) }
  end

  it "lists cpu and blas first" do
    Autoresearch.device_names.first(2).should eq ["cpu", "blas"]
  end
end
