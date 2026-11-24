# Prepares the data and tokenizer for autoresearch, as prepare.py does, then
# encodes the data into token shards for training. Each step skips work
# already done, so an interrupted run resumes.
#
# Usage: bin/prepare [command] [--num-shards N] [--download-workers N]
#
# Commands, run in this order by `all` (the default): `fetch` downloads the
# Parquet shards, `train-tokenizer` trains the tokenizer, `tokenize` writes
# the token shards. Everything goes under the cache directory,
# `AUTORESEARCH_CACHE` or `~/.cache/autoresearch.cr`.
require "option_parser"
require "./prepare/fetch"
require "./prepare/train_tokenizer"
require "./prepare/tokenize"

command = "all"
num_shards = 10
workers = 8
OptionParser.parse do |parser|
  parser.banner = "Usage: bin/prepare [all|fetch|train-tokenizer|tokenize] [options]"
  parser.on("--num-shards N", "Training shards to download (-1 = all); the validation shard is always fetched (default 10)") do |value|
    num_shards = value.to_i == -1 ? Autoresearch::MAX_SHARD : value.to_i
  end
  parser.on("--download-workers N", "Parallel downloads (default 8)") { |value| workers = value.to_i }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
  parser.unknown_args { |args| command = args.first? || command }
end

puts "Cache directory: #{Autoresearch.cache_dir}"
puts
case command
when "fetch"           then Prepare::Fetch.run(num_shards, workers)
when "train-tokenizer" then Prepare::TrainTokenizer.run
when "tokenize"        then Prepare::Tokenize.run
when "all"
  Prepare::Fetch.run(num_shards, workers)
  puts
  Prepare::TrainTokenizer.run
  puts
  Prepare::Tokenize.run
  puts
  puts "Done! Ready to train."
else
  abort "unknown command #{command}: use all, fetch, train-tokenizer or tokenize"
end
