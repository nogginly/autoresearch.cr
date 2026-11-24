require "../../src/autoresearch/constants"
require "../../src/autoresearch/tokenizer"
require "./parquet"
require "./bpe_trainer"
require "./tokenizer_files"

module Prepare
  # Trains the tokenizer as prepare.py's `train_tokenizer` does, and writes
  # `tokenizer.json` and `token_bytes.json`; skipped when both exist.
  #
  # The training text is the training shards' documents in order (every
  # shard but the validation one, by name, row group by row group), each cut
  # to `DOC_CAP` characters, until `MAX_CHARS` characters have been read: the
  # document that crosses the limit is the last. Training finds
  # `VOCAB_SIZE - 4` tokens, and the four special tokens follow.
  module TrainTokenizer
    MAX_CHARS = 1_000_000_000_i64
    DOC_CAP   =            10_000

    # Yields the training documents, as prepare.py's `text_iterator` does.
    def self.each_document(dir : String, max_chars : Int64 = MAX_CHARS, doc_cap : Int32 = DOC_CAP, & : String ->) : Nil
      val = Autoresearch.shard_name(Autoresearch::VAL_SHARD, "parquet")
      paths = Dir.children(dir).select { |name| name.ends_with?(".parquet") && name != val }.sort!
      chars = 0_i64
      paths.each do |name|
        file = Parquet::File.new(File.join(dir, name))
        file.row_groups.times do |group|
          file.strings(group, "text").each do |text|
            document = text.size > doc_cap ? text[0, doc_cap] : text
            chars += document.size
            yield document
            return if chars >= max_chars
          end
        end
      end
    end

    def self.run(data_dir : String = Autoresearch.data_dir, dir : String = Autoresearch.tokenizer_dir) : Nil
      json_path = File.join(dir, "tokenizer.json")
      bytes_path = File.join(dir, "token_bytes.json")
      if File.exists?(json_path) && File.exists?(bytes_path)
        puts "Tokenizer: already trained at #{dir}"
        return
      end
      parquet = Dir.exists?(data_dir) ? Dir.children(data_dir).count(&.ends_with?(".parquet")) : 0
      abort "Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first." if parquet < 2
      Dir.mkdir_p(dir)

      puts "Tokenizer: training BPE tokenizer..."
      started = Time.instant
      trainer = BPETrainer.new(Regex.new(Autoresearch::SPLIT_PATTERN, Regex::Options::UCP))
      each_document(data_dir) { |document| trainer.add(document) }
      merges = trainer.train(Autoresearch::VOCAB_SIZE - Autoresearch::SPECIAL_TOKENS.size - 256)
      TokenizerFiles.write_json("#{json_path}.tmp", merges, Autoresearch::SPECIAL_TOKENS, Autoresearch::SPLIT_PATTERN)
      File.write("#{bytes_path}.tmp", TokenizerFiles.byte_lengths(merges, Autoresearch::SPECIAL_TOKENS).to_json)
      File.rename("#{bytes_path}.tmp", bytes_path)
      File.rename("#{json_path}.tmp", json_path)
      puts "Tokenizer: trained in #{(Time.instant - started).total_seconds.round(1)}s, saved to #{dir}"

      tokenizer = Autoresearch::Tokenizer.from_directory(dir)
      test = "Hello world! Numbers: 123. Unicode: 你好"
      decoded = tokenizer.decode(tokenizer.encode(test))
      abort "Tokenizer: roundtrip failed: #{test.inspect} -> #{decoded.inspect}" unless decoded == test
      puts "Tokenizer: sanity check passed (vocab_size=#{tokenizer.vocab_size})"
    end
  end
end
