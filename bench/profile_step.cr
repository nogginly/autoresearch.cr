# Profiles training steps on a GPU and prints tensomite's per-op report: three
# warm-up steps, *steps* steps timed as they run, then *steps* more with each op
# timed alone on the GPU.
#
# Usage: crystal run --release -Dautoresearch_spec bench/profile_step.cr -- DEVICE [batch] [steps] [sequence] [depth]
# (`-` for *sequence* or *depth* keeps its default)
# (`make profile DEVICE=… BATCH=… STEPS=… SEQ=… DEPTH=…`), where DEVICE is metal
# (Apple Silicon) or cuda (a `-Dcuda` build, `CUDA=1`).
#
# A training step is one micro-batch of *batch* sequences: forward, backward and
# the MuonAdamW step. *sequence* defaults to `MAX_SEQ_LEN` and *depth* to `DEPTH`.
require "./common"

def profile_training(gpu, normal, timed, label : String, count : Int32, &) : NoReturn
  3.times { yield }
  gpu.profile = normal
  start = Time.instant
  count.times { yield }
  wall = (Time.instant - start).total_seconds
  gpu.profile = timed
  count.times { yield }
  gpu.profile = nil
  normal.class.report(STDOUT, gpu.name, label, count, wall, normal, timed, top: 30)
  exit
end

device = pick_device(ARGV[0]? || "blas")
batch = (ARGV[1]? || "8").to_i
count = (ARGV[2]? || "1").to_i
sequence = ARGV[3]?.try { |arg| arg == "-" ? nil : arg.to_i } || Autoresearch::MAX_SEQ_LEN
depth = ARGV[4]?.try { |arg| arg == "-" ? nil : arg.to_i } || DEPTH
config = bench_config(depth, sequence)
model, optimizer, idx, targets = bench_setup(config, device, batch)
label = "training step (depth #{config.n_layer}, #{batch} × #{config.sequence_len} tokens)"
puts "model: #{config}"

train_step = -> do
  apply_schedules(optimizer, 0.0, optimizer.steps)
  optimizer.step(model.call(idx, targets))
  GC.collect
  nil
end

{% if flag?(:cuda) %}
  if cuda = device.as?(CUDA)
    profile_training(cuda, CUDA::Profile.new, CUDA::Profile.new(per_kernel: true, shapes: true), label, count) { train_step.call }
  end
{% end %}
{% if flag?(:darwin) && flag?(:aarch64) %}
  if metal = device.as?(Metal)
    profile_training(metal, Metal::Profile.new, Metal::Profile.new(per_kernel: true, shapes: true), label, count) { train_step.call }
  end
{% end %}
abort "profiling needs DEVICE=metal on Apple Silicon or DEVICE=cuda in a CUDA=1 build"
