require "digest/sha256"
require "file_utils"
require "../../tools/prepare/parquet"
require "../../tools/prepare/bpe_trainer"
require "../../tools/prepare/tokenizer_files"
require "../../tools/prepare/train_tokenizer"
require "../../tools/prepare/tokenize"
require "../../src/autoresearch/dataloader"

# Fixtures from scripts/fixtures/make_fixtures.py: a small dataset in the
# climbmix shards' layout, and what prepare.py's own train_tokenizer,
# Tokenizer and make_dataloader make of it (see spec/fixtures/README.md).
DATA_FIXTURES = "#{__DIR__}/../fixtures/data"

def data_meta : JSON::Any
  JSON.parse(File.read("#{DATA_FIXTURES}/expected.json"))
end

def int32_sha(ids : Array(Int32)) : String
  io = IO::Memory.new
  ids.each { |id| io.write_bytes(id, IO::ByteFormat::LittleEndian) }
  Digest::SHA256.hexdigest(io.to_slice)
end

# Returns the merges BPETrainer finds on the fixture's training shards, as prepare.py's train_tokenizer would.
def fixture_merges : Array({Int32, Int32})
  trainer = BPETrainer.new(Regex.new(Autoresearch::SPLIT_PATTERN, Regex::Options::UCP))
  Prepare::TrainTokenizer.each_document("#{DATA_FIXTURES}/parquet") { |document| trainer.add(document) }
  trainer.train(Autoresearch::VOCAB_SIZE - Autoresearch::SPECIAL_TOKENS.size - 256)
end

# Runs *block* with a scratch directory holding the fixture's tokenizer and token shards, as bin/prepare makes them.
def with_prepared(& : String, Autoresearch::Tokenizer ->) : Nil
  dir = File.tempname("autoresearch_data")
  Dir.mkdir_p(dir)
  begin
    tokenizer_dir = File.join(dir, "tokenizer")
    Dir.mkdir_p(tokenizer_dir)
    merges = fixture_merges
    TokenizerFiles.write_json(File.join(tokenizer_dir, "tokenizer.json"), merges, Autoresearch::SPECIAL_TOKENS, Autoresearch::SPLIT_PATTERN)
    File.write(File.join(tokenizer_dir, "token_bytes.json"), TokenizerFiles.byte_lengths(merges, Autoresearch::SPECIAL_TOKENS).to_json)
    tokenizer = Autoresearch::Tokenizer.from_directory(tokenizer_dir)
    tokens_dir = File.join(dir, "tokens")
    Prepare::Tokenize.run("#{DATA_FIXTURES}/parquet", tokens_dir, tokenizer, log: IO::Memory.new)
    yield tokens_dir, tokenizer
  ensure
    FileUtils.rm_rf(dir)
  end
end
