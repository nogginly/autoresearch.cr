# Fixed constants of the harness, as in `prepare.py`. The agent does not edit this file.
module Autoresearch
  # The profile's name, and its context length in tokens: `small`'s 512 by
  # default, `reference`'s 2,048 with `-Dreference` (PLAN.md, Decision H).
  {% if flag?(:reference) %}
    MAX_SEQ_LEN = 2048
    PROFILE     = "reference"
  {% else %}
    MAX_SEQ_LEN = 512
    PROFILE     = "small"
  {% end %}

  # Training time budget, in seconds, not counting start-up.
  TIME_BUDGET = 300

  # Validation tokens `evaluate_bpb` reads.
  EVAL_TOKENS = 40 * 524288

  # Tokenizer vocabulary, special tokens included.
  VOCAB_SIZE = 8192

  # The pattern that splits text into chunks before byte-pair encoding: GPT-4's,
  # with numbers of at most two digits.
  SPLIT_PATTERN = %q('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)

  # Special tokens, which take the ids after the merged tokens; the first begins every document.
  SPECIAL_TOKENS = (0...4).map { |i| "<|reserved_#{i}|>" }
  BOS_TOKEN      = SPECIAL_TOKENS.first

  # Where the dataset's Parquet shards are downloaded from.
  BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"

  # The last shard's index; it is the pinned validation shard.
  MAX_SHARD = 6542
  VAL_SHARD = MAX_SHARD

  # The cache: `AUTORESEARCH_CACHE`, or `~/.cache/autoresearch.cr`.
  def self.cache_dir : String
    ENV["AUTORESEARCH_CACHE"]? || Path.home.join(".cache", "autoresearch.cr").to_s
  end

  # Downloaded Parquet shards.
  def self.data_dir : String
    File.join(cache_dir, "data")
  end

  # `tokenizer.json` and `token_bytes.json`.
  def self.tokenizer_dir : String
    File.join(cache_dir, "tokenizer")
  end

  # Token shards, one per Parquet shard.
  def self.tokens_dir : String
    File.join(cache_dir, "tokens")
  end

  # Returns the file name of shard *index*, `shard_00042` and so on, with *extension*.
  def self.shard_name(index : Int32, extension : String) : String
    "shard_#{index.to_s.rjust(5, '0')}.#{extension}"
  end
end

module Autoresearch
  # Returns the prepared files training needs that are missing under *cache*: the tokenizer, the
  # validation token shard and at least one training token shard. Empty when `bin/prepare` has finished.
  def self.missing_prepared(cache : String = cache_dir) : Array(String)
    tokenizer = File.join(cache, "tokenizer")
    tokens = File.join(cache, "tokens")
    val = shard_name(VAL_SHARD, "tokens")
    missing = %w[tokenizer.json token_bytes.json].map { |name| File.join(tokenizer, name) }.reject { |path| File.exists?(path) }
    missing << File.join(tokens, val) unless File.exists?(File.join(tokens, val))
    train = Dir.exists?(tokens) && Dir.children(tokens).any? { |name| name.ends_with?(".tokens") && name != val }
    missing << File.join(tokens, "shard_*.tokens (training)") unless train
    missing
  end
end
