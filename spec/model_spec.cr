require "./spec_helper"

# Fixtures from scripts/fixtures/make_model_fixtures.py: train.py's model at the
# pinned commit, run by PyTorch on the CPU in float32 (see spec/fixtures/README.md).
MODEL_FIXTURES = "#{__DIR__}/fixtures/model"

# Linear weights, which PyTorch stores as [out, in] and tensomite as [in, out].
TRANSPOSED = /(c_q|c_k|c_v|c_proj|c_fc|ve_gate|lm_head)\.weight$/

# Every comparison allows an error of 2e-5 times the tensor's largest magnitude.
# BLAS against PyTorch's CPU path measured at most 2.3e-6 (the x0_lambdas
# gradient), from summing in a different order; an off-by-one window, a wrong
# RoPE sign or a wrong layer's scalar measured 0.03 to 1.5.
PARITY = 2e-5

def model_meta : JSON::Any
  JSON.parse(File.read("#{MODEL_FIXTURES}/expected.json"))
end

def tiny_config : GPTConfig
  t = model_meta["tiny"]
  GPTConfig.new(
    sequence_len: t["sequence_len"].as_i, vocab_size: t["vocab_size"].as_i, n_layer: t["n_layer"].as_i,
    n_head: t["n_head"].as_i, n_kv_head: t["n_kv_head"].as_i, n_embd: t["n_embd"].as_i,
    window_pattern: t["window_pattern"].as_s,
  )
end

def fixture_model(device : Device) : GPT
  GPT.new(tiny_config, device).load("#{MODEL_FIXTURES}/weights.safetensors", transposed: TRANSPOSED)
end

def fixture_inputs(device : Device) : {Tensor, Tensor}
  inputs = SafeTensors.open("#{MODEL_FIXTURES}/inputs.safetensors")
  {inputs["idx", device], inputs["targets", device]}
end

def expected(name : String) : Array(Float32)
  SafeTensors.open("#{MODEL_FIXTURES}/expected.safetensors")[name].to_a(Float32)
end

describe "M1 exit: the model matches train.py" do
  MODEL_DEVICES.each do |device|
    describe "on #{device.class.name}" do
      it "gives the same logits" do
        idx, _ = fixture_inputs(device)
        expect_close_to_scale(fixture_model(device).call(idx).to_a(Float32), expected("logits"), PARITY, "logits")
      end

      it "gives the same loss, ignoring targets of -1" do
        idx, targets = fixture_inputs(device)
        expect_close_to_scale(fixture_model(device).call(idx, targets).to_a(Float32), expected("loss"), PARITY, "loss")
      end

      it "gives the same per-token losses, 0 where the target is -1" do
        idx, targets = fixture_inputs(device)
        losses = fixture_model(device).call(idx, targets, Reduction::None).to_a(Float32)
        expect_close_to_scale(losses, expected("per_token_loss"), PARITY, "per_token_loss")
        losses[3].should eq 0f32
      end

      it "gives the same gradient for every parameter" do
        idx, targets = fixture_inputs(device)
        model = fixture_model(device)
        named = model.named_params.to_a
        grads = Tensomite.grad(model.call(idx, targets), named.map(&.[1].value))
        named.each_with_index do |(name, _), i|
          grad = name.matches?(TRANSPOSED) ? grads[i].transpose.contiguous : grads[i]
          expect_close_to_scale(grad.to_a(Float32), expected("grad.#{name}"), PARITY, "grad.#{name}")
        end
      end
    end
  end

  it "has the same parameters, by name, as the fixture" do
    names = model_meta["names"].as_h.values.map(&.as_s).sort!
    GPT.new(tiny_config, DEVICE).named_params.keys.sort!.should eq names
  end

  it "has the same attention windows" do
    GPT.new(tiny_config, DEVICE).window_sizes.should eq model_meta["window_sizes"].as_a.map(&.as_i)
  end

  it "counts the same parameters as train.py at depth 8" do
    model = GPT.new(build_model_config(8), DEVICE)
    expected = model_meta["depth8"]["num_scaling_params"].as_h.transform_values(&.as_i64)
    model.num_scaling_params.should eq expected
  end

  it "estimates the same FLOPs per token as train.py at depth 8" do
    model = GPT.new(build_model_config(8), DEVICE)
    model.estimate_flops.should eq model_meta["depth8"]["estimate_flops"].as_i64
  end

  it "builds train.py's depth-8 config" do
    c = model_meta["depth8"]["config"]
    config = build_model_config(8)
    {config.sequence_len, config.vocab_size, config.n_layer, config.n_head, config.n_kv_head, config.n_embd, config.window_pattern}
      .should eq({c["sequence_len"].as_i, c["vocab_size"].as_i, c["n_layer"].as_i, c["n_head"].as_i, c["n_kv_head"].as_i, c["n_embd"].as_i, c["window_pattern"].as_s})
  end
end

describe GPT do
  it "starts from train.py's initial values" do
    model = GPT.new(tiny_config, DEVICE)
    model.init_weights(Random.new(1))
    params = model.named_params
    params["resid_lambdas"].value.to_a(Float32).should eq [1f32] * 4
    params["x0_lambdas"].value.to_a(Float32).should eq [0.1f32] * 4
    {"h.0.attn.c_proj.weight", "h.0.mlp.c_proj.weight", "h.3.attn.ve_gate.weight"}.each do |name|
      params[name].value.to_a(Float32).all?(&.zero?).should be_true
    end
    # Uniform ±√3/√64 ≈ ±0.2165
    params["h.0.attn.c_q.weight"].value.to_a(Float32).max_of(&.abs).should be <= Math.sqrt(3.0) / 8
  end

  it "rejects grouped-query attention, which is not ported" do
    expect_raises(ArgumentError, /grouped-query/) do
      GPT.new(GPTConfig.new(sequence_len: 16, vocab_size: 128, n_layer: 2, n_head: 2, n_kv_head: 1, n_embd: 64), DEVICE)
    end
  end
end
