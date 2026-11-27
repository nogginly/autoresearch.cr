require "./spec_helper"
require "./support/data_fixtures"
require "../src/autoresearch/evaluate"

# Fixture from scripts/fixtures/make_fixtures.py: prepare.py's own
# evaluate_bpb of a tiny model on the data fixture's validation shard (see
# spec/fixtures/README.md).
EVAL_FIXTURES = "#{__DIR__}/fixtures/eval"

def eval_meta : JSON::Any
  JSON.parse(File.read("#{EVAL_FIXTURES}/expected.json"))
end

def eval_model(device : Device) : GPT
  c = eval_meta["config"]
  config = GPTConfig.new(
    sequence_len: c["sequence_len"].as_i, vocab_size: c["vocab_size"].as_i, n_layer: c["n_layer"].as_i,
    n_head: c["n_head"].as_i, n_kv_head: c["n_kv_head"].as_i, n_embd: c["n_embd"].as_i,
    window_pattern: c["window_pattern"].as_s,
  )
  GPT.new(config, device).load("#{EVAL_FIXTURES}/weights.safetensors", transposed: TRANSPOSED)
end

describe "M4 exit: evaluate_bpb matches prepare.py's" do
  MODEL_DEVICES.each do |device|
    # Four significant digits, as the exit asks.
    it "gives the fixture's val_bpb to four significant digits on #{device.class.name}" do
      with_prepared do |tokens_dir, tokenizer|
        ours = Autoresearch.evaluate_bpb(eval_model(device), tokenizer, eval_meta["batch_size"].as_i, device,
          dir: tokens_dir, max_seq_len: eval_meta["max_seq_len"].as_i, eval_tokens: eval_meta["eval_tokens"].as_i)
        theirs = eval_meta["val_bpb"].as_f
        ours.should be_close(theirs, 5e-5 * theirs)
      end
    end
  end
end
