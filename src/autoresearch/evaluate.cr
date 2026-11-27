require "tensomite"
require "./constants"
require "./dataloader"
require "./tokenizer"

module Autoresearch
  # Returns bits per byte on the validation split, the fixed metric, as
  # prepare.py's `evaluate_bpb` computes it:
  #
  # 1. Reads `eval_tokens // (batch_size × max_seq_len)` batches of the
  #    validation token shards in *dir*, packed as for training.
  # 2. Sums the per-token losses, in nats, over targets whose byte length is
  #    above zero (special tokens count 0), and sums those byte lengths.
  # 3. Returns `nats / (ln 2 × bytes)`.
  #
  # *model* is anything whose `call(idx, targets, Tensomite::Reduction::None)`
  # returns one loss per target, such as train.cr's `GPT`; it runs on *device*.
  def self.evaluate_bpb(model, tokenizer : Tokenizer, batch_size : Int32, device : Tensomite::Device,
                        dir : String = tokens_dir, max_seq_len : Int32 = MAX_SEQ_LEN,
                        eval_tokens : Int32 = EVAL_TOKENS) : Float64
    loader = Dataloader.new(Dataloader.split("val", dir), batch_size, max_seq_len)
    steps = eval_tokens // (batch_size * max_seq_len)
    shape = [batch_size, max_seq_len]
    total_nats = 0.0
    total_bytes = 0_i64
    steps.times do
      inputs, targets, _ = loader.next_batch
      nbytes = targets.map { |id| tokenizer.token_bytes[id] }
      mask = Tensomite::Tensor.from(nbytes.map { |size| size > 0 ? 1f32 : 0f32 }, [nbytes.size], device)
      losses = model.call(Tensomite::Tensor.from(inputs, shape, device), Tensomite::Tensor.from(targets, shape, device),
        Tensomite::Reduction::None)
      nats = (losses * mask).sum
      Tensomite::Tensor.realize([nats], release: true)
      total_nats += nats.to_a(Float32).first
      total_bytes += nbytes.sum(0_i64)
      GC.collect
    end
    total_nats / (Math.log(2.0) * total_bytes)
  end
end
