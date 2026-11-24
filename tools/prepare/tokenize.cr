require "../../src/autoresearch/constants"
require "../../src/autoresearch/shards"
require "../../src/autoresearch/tokenizer"
require "./parquet"

module Prepare
  # Encodes every downloaded Parquet shard into a token shard (see
  # `Autoresearch::TokenShard`): each document's ids, led by BOS, as
  # prepare.py's dataloader encodes them. A token shard already present is
  # skipped.
  module Tokenize
    def self.run(data_dir : String = Autoresearch.data_dir, dir : String = Autoresearch.tokens_dir,
                 tokenizer : Autoresearch::Tokenizer = Autoresearch::Tokenizer.from_directory, log : IO = STDOUT) : Nil
      Dir.mkdir_p(dir)
      names = Dir.children(data_dir).select(&.ends_with?(".parquet")).sort!
      todo = names.reject { |name| File.exists?(File.join(dir, name.sub(/\.parquet$/, ".tokens"))) }
      if todo.empty?
        log.puts "Tokens: all #{names.size} shards already tokenized at #{dir}"
        return
      end
      log.puts "Tokens: tokenizing #{todo.size} shards (#{names.size - todo.size} already done)..."
      todo.each do |name|
        started = Time.instant
        file = Parquet::File.new(File.join(data_dir, name))
        documents = [] of Array(Int32)
        sizes = Array.new(file.row_groups) do |group|
          texts = file.strings(group, "text")
          texts.each { |text| documents << tokenizer.encode(text, bos: true) }
          texts.size
        end
        destination = File.join(dir, name.sub(/\.parquet$/, ".tokens"))
        Autoresearch::TokenShard.write(destination, sizes, documents)
        tokens = documents.sum(&.size)
        log.puts "  Tokenized #{name}: #{documents.size} documents, #{tokens} tokens, #{(Time.instant - started).total_seconds.round(1)}s"
      end
    end
  end
end
