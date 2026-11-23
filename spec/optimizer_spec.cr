require "./spec_helper"

# Fixtures from scripts/fixtures/make_fixtures.py: five steps of train.py's
# MuonAdamW on the tiny model from fixed gradients (see spec/fixtures/README.md).
OPTIMIZER_FIXTURES = "#{__DIR__}/fixtures/optimizer"

# Each comparison is of the update, the change from the initial parameter, and
# allows an error of 1e-4 times the update's largest magnitude. BLAS against
# PyTorch's CPU path measured at most 8.6e-6 (lm_head after one step, where
# Adam divides by a second moment near zero). Breaking NorMuon's axis, Nesterov
# momentum, cautious decay, the tall-matrix rate, Adam's bias correction, the
# momentum schedule or the number of Newton–Schulz steps measured 0.11 to 0.92.
UPDATE_PARITY = 1e-4

# Cautious weight decay masks each element by the sign of `g × p`, a step
# function. Where that product is within a rounding error of zero, as one
# element of h.0's c_fc is at step 3 (1.2e-8), another device's last bit can
# flip the mask and move the element by `lr × wd × p`. A Muon parameter's
# update may therefore have up to this fraction of elements outside
# `UPDATE_PARITY`, each by no more than the sum over the steps of
# `lr × wd × max|p|`. Breaking the mask itself moves about half of them.
MASK_FLIP_FRACTION = 1e-3

def optimizer_meta : JSON::Any
  JSON.parse(File.read("#{OPTIMIZER_FIXTURES}/expected.json"))
end

def optimizer_tensors : SafeTensors
  SafeTensors.open("#{OPTIMIZER_FIXTURES}/tensors.safetensors")
end

def fixture_optimizer(model : GPT) : MuonAdamW
  model.setup_optimizer(unembedding_lr: UNEMBEDDING_LR, embedding_lr: EMBEDDING_LR, scalar_lr: SCALAR_LR,
    adam_betas: ADAM_BETAS, matrix_lr: MATRIX_LR, weight_decay: WEIGHT_DECAY)
end

# Takes fixture steps *from* to *to* (1-based, inclusive) from *values*, and returns the new values.
def run_steps(optimizer : MuonAdamW, model : GPT, values : Array(Tensor), from : Int32, to : Int32, device : Device) : Array(Tensor)
  names = model.named_params.keys
  tensors = optimizer_tensors
  (from..to).each do |k|
    step = optimizer_meta["steps"][k - 1]
    apply_schedules(optimizer, step["progress"].as_f, step["step"].as_i)
    grads = names.map { |name| other_layout(name, tensors["grad.#{k}.#{name}", device]) }
    values = optimizer.step(values, grads)
  end
  values
end

# Returns the largest change a flip of the cautious mask can make to a Muon
# parameter of PyTorch *shape*, whose largest magnitude is *max_param*, over
# the first *after* fixture steps: the sum of `lr × wd × max|p|`.
def mask_flip_bound(shape : Array(Int32), max_param : Float64, after : Int32) : Float64
  rows, cols = shape[0], shape[1]
  optimizer_meta["steps"].as_a.first(after).sum do |step|
    lr = MATRIX_LR * step["lr_multiplier"].as_f * Math.max(1.0, rows / cols) ** 0.5
    lr * step["muon_weight_decay"].as_f * max_param
  end
end

def expect_updates(model : GPT, values : Array(Tensor), after : Int32) : Nil
  tensors = optimizer_tensors
  muon = model.named_params.keys.select(&.starts_with?("transformer.h."))
  model.named_params.keys.each_with_index do |name, i|
    initial_tensor = tensors["initial.#{name}"]
    initial = initial_tensor.to_a(Float32)
    ours = other_layout(name, values[i]).to_a(Float32).zip(initial).map { |(a, b)| a - b }
    theirs = tensors["after#{after}.#{name}"].to_a(Float32).zip(initial).map { |(a, b)| a - b }
    label = "update of #{name} after #{after}"
    if muon.includes?(name)
      bound = mask_flip_bound(initial_tensor.shape, initial.max_of(&.abs).to_f64, after)
      expect_close_allowing_flips(ours, theirs, UPDATE_PARITY, MASK_FLIP_FRACTION, bound, label)
    else
      expect_close_to_scale(ours, theirs, UPDATE_PARITY, label)
    end
  end
end

# Checks *actual* against *expected* as `expect_close_to_scale` does, except
# that up to *fraction* of the elements may differ by as much as *flip_bound*.
def expect_close_allowing_flips(actual : Array(Float32), expected : Array(Float32), relative : Float64,
                                fraction : Float64, flip_bound : Float64, label : String) : Nil
  actual.size.should eq expected.size
  limit = relative * expected.max_of(&.abs).to_f64
  outliers = actual.zip(expected).each_with_index.select { |(a, e), _| (a - e).abs > limit }.to_a
  allowed = Math.max(1, (fraction * actual.size).to_i)
  if outliers.size > allowed
    fail "#{label}: #{outliers.size} of #{actual.size} elements differ by more than #{limit}; at most #{allowed} may"
  end
  outliers.each do |(a, e), index|
    fail "#{label}: element #{index} is #{a}, expected #{e}; more than a mask flip (#{flip_bound})" if (a - e).abs > flip_bound + limit
  end
end

describe "M2 exit: MuonAdamW matches train.py" do
  MODEL_DEVICES.each do |device|
    describe "on #{device.class.name}" do
      it "updates every parameter as train.py does after 1 and 5 steps" do
        model = fixture_model(device)
        optimizer = fixture_optimizer(model)
        values = run_steps(optimizer, model, model.params.map(&.value), 1, 1, device)
        expect_updates(model, values, after: 1)
        values = run_steps(optimizer, model, values, 2, 5, device)
        expect_updates(model, values, after: 5)
      end

      it "resumes from saved state exactly as an uninterrupted run" do
        model = fixture_model(device)
        optimizer = fixture_optimizer(model)
        straight = run_steps(optimizer, model, model.params.map(&.value), 1, 5, device)

        model = fixture_model(device)
        optimizer = fixture_optimizer(model)
        values = run_steps(optimizer, model, model.params.map(&.value), 1, 2, device)
        path = File.tempname("muon_adamw", ".safetensors")
        begin
          SafeTensors.write(path, optimizer.state)
          resumed = fixture_optimizer(fixture_model(device)).restore(SafeTensors.open(path), steps: optimizer.steps)
          values = run_steps(resumed, model, values, 3, 5, device)
        ensure
          File.delete?(path)
        end
        values.map(&.to_a(Float32)).should eq straight.map(&.to_a(Float32))
      end
    end
  end

  it "uses train.py's hyperparameters" do
    h = optimizer_meta["hyperparameters"]
    {EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR, SCALAR_LR, WEIGHT_DECAY, ADAM_BETAS[0], ADAM_BETAS[1], WARMUP_RATIO, WARMDOWN_RATIO, FINAL_LR_FRAC}
      .should eq({h["EMBEDDING_LR"].as_f, h["UNEMBEDDING_LR"].as_f, h["MATRIX_LR"].as_f, h["SCALAR_LR"].as_f, h["WEIGHT_DECAY"].as_f,
                  h["ADAM_BETAS"][0].as_f, h["ADAM_BETAS"][1].as_f, h["WARMUP_RATIO"].as_f, h["WARMDOWN_RATIO"].as_f, h["FINAL_LR_FRAC"].as_f})
  end

  it "groups parameters as train.py's setup_optimizer does" do
    model = fixture_model(DEVICE)
    names = model.named_params.keys
    groups = fixture_optimizer(model).param_groups
    expected = optimizer_meta["groups"].as_a
    groups.size.should eq expected.size
    groups.zip(expected) do |group, theirs|
      group.params.map { |i| names[i] }.should eq theirs["params"].as_a.map(&.as_s)
      group.lr.should be_close(theirs["lr"].as_f, 1e-12)
      group.weight_decay.should eq theirs["weight_decay"].as_f
      case group
      in AdamWGroup
        theirs["kind"].should eq "adamw"
        [group.betas[0], group.betas[1]].should eq theirs["betas"].as_a.map(&.as_f)
        group.eps.should eq theirs["eps"].as_f
      in MuonGroup
        theirs["kind"].should eq "muon"
        group.momentum.should eq theirs["momentum"].as_f
        group.ns_steps.should eq theirs["ns_steps"].as_i
        group.beta2.should eq theirs["beta2"].as_f
      end
    end
  end

  it "schedules the learning rate, momentum and weight decay as train.py does" do
    optimizer_meta["steps"].as_a.each do |step|
      progress = step["progress"].as_f
      get_lr_multiplier(progress).should be_close(step["lr_multiplier"].as_f, 1e-12)
      get_muon_momentum(step["step"].as_i).should be_close(step["muon_momentum"].as_f, 1e-12)
      get_weight_decay(progress).should be_close(step["muon_weight_decay"].as_f, 1e-12)
    end
  end
end

describe MuonAdamW do
  it "rejects groups that leave a parameter out" do
    model = fixture_model(DEVICE)
    groups = fixture_optimizer(model).param_groups[1..]
    expect_raises(ArgumentError, /each of/) { MuonAdamW.new(model.params, groups) }
  end
end
