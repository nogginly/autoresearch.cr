require "./constants"
require "./shards"

module Autoresearch
  # Batches of *b* rows of *t* tokens from token shards, packed as prepare.py's
  # `make_dataloader` packs them: every row starts with a document's BOS, and
  # documents fill rows best-fit, without padding.
  #
  # Documents enter a buffer of at least *buffer_size* documents, in batches
  # of up to 128 that never cross a Parquet row group, shard after shard; when
  # the shards run out, a new epoch starts from the first. Each row of `t + 1`
  # tokens is filled by repeatedly taking the longest buffered document that
  # fits the space left, the first such on ties; when none fits, the shortest
  # is cut to fill it and the rest of it dropped. Inputs are each row's first
  # `t` tokens, targets its last `t`.
  #
  # ```
  # loader = Dataloader.new(Dataloader.split("train"), b: 8, t: 2048)
  # inputs, targets, epoch = loader.next_batch # each b × t ids, row-major
  # ```
  class Dataloader
    TOKENIZER_BATCH_SIZE = 128

    # The epoch of the documents most recently added to the buffer.
    getter epoch = 1

    def initialize(@shards : Array(TokenShard), @b : Int32, @t : Int32, @buffer_size : Int32 = 1000)
      raise ArgumentError.new("no token shards") if @shards.empty?
      @buffer = [] of Slice(UInt16)
      @shard = 0
      @group = 0
      @next_document = 0
      @group_start = 0
      @cursor_epoch = 1
    end

    # Returns the token shards of *split*, `"train"` (every shard in *dir* but
    # the validation shard, in name order) or `"val"` (the validation shard).
    def self.split(split : String, dir : String = Autoresearch.tokens_dir) : Array(TokenShard)
      val = Autoresearch.shard_name(VAL_SHARD, "tokens")
      names = Dir.children(dir).select(&.ends_with?(".tokens")).sort!
      chosen = case split
               when "train" then names.reject(&.==(val))
               when "val"   then names.select(&.==(val))
               else              raise ArgumentError.new("split must be train or val, not #{split}")
               end
      raise Tensomite::Error.new("no #{split} token shards in #{dir}; run bin/prepare") if chosen.empty?
      chosen.map { |name| TokenShard.new(File.join(dir, name)) }
    end

    # Returns the next batch: inputs and targets, each `b × t` ids in
    # row-major order, and the epoch of the last documents read.
    def next_batch : {Array(Int32), Array(Int32), Int32}
      capacity = @t + 1
      rows = Array(Int32).new(@b * capacity)
      @b.times do
        pos = 0
        while pos < capacity
          while @buffer.size < @buffer_size
            refill
          end
          remaining = capacity - pos
          best = -1
          best_size = 0
          @buffer.each_with_index do |document, i|
            if document.size <= remaining && document.size > best_size
              best = i
              best_size = document.size
            end
          end
          if best >= 0
            @buffer.delete_at(best).each { |id| rows << id.to_i32 }
            pos += best_size
          else
            shortest = (0...@buffer.size).min_by { |i| @buffer[i].size }
            @buffer.delete_at(shortest)[0, remaining].each { |id| rows << id.to_i32 }
            pos += remaining
          end
        end
      end
      inputs = Array(Int32).new(@b * @t)
      targets = Array(Int32).new(@b * @t)
      @b.times do |row|
        start = row * capacity
        @t.times do |i|
          inputs << rows[start + i]
          targets << rows[start + i + 1]
        end
      end
      {inputs, targets, @epoch}
    end

    # Adds the next batch of documents to the buffer: up to
    # `TOKENIZER_BATCH_SIZE` from the current row group, moving to the next
    # row group, shard and, after the last shard, epoch as each runs out.
    private def refill : Nil
      @epoch = @cursor_epoch
      shard = @shards[@shard]
      group_end = @group_start + shard.row_group_sizes[@group]
      stop = Math.min(@next_document + TOKENIZER_BATCH_SIZE, group_end)
      (@next_document...stop).each { |i| @buffer << shard.document(i) }
      @next_document = stop
      return if stop < group_end
      @group += 1
      @group_start = group_end
      return if @group < shard.row_group_sizes.size
      @group = @group_start = @next_document = 0
      @shard += 1
      return if @shard < @shards.size
      @shard = 0
      @cursor_epoch += 1
    end
  end
end
