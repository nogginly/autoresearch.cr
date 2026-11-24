# Byte-pair-encoding training as rustbpe 0.1.0 does it, the trainer
# prepare.py uses.
#
# Text is split into chunks by a pattern, and identical chunks are counted.
# Each chunk starts as its UTF-8 bytes, ids 0 to 255. Each merge then takes the
# adjacent pair with the highest count over all chunks, weighted by how often
# each chunk occurs (ties go to the smaller pair, comparing left ids, then
# right), gives it the next id from 256, and replaces its occurrences in every
# chunk, left to right without overlap.
#
# ```
# trainer = BPETrainer.new(Regex.new(SPLIT_PATTERN, Regex::Options::UCP))
# texts.each { |text| trainer.add(text) }
# merges = trainer.train(8188 - 256) # [{left, right}, ...], the merge for id 256 first
# ```
class BPETrainer
  alias Pair = {Int32, Int32}

  # The chunk counts so far, by chunk.
  getter chunks = Hash(String, Int32).new(0)

  def initialize(@pattern : Regex)
  end

  # Splits *text* with the pattern and counts its chunks.
  def add(text : String) : Nil
    text.scan(@pattern) { |match| @chunks[match[0]] += 1 }
  end

  # Returns up to *count* merges, in order: the pair merged into id `256 + i`
  # at index `i`. Fewer when no pair is left to merge.
  #
  # 1. Counts every adjacent pair, and notes which chunks hold it.
  # 2. Repeatedly takes the best pair from a priority queue, re-queueing it
  #    first if its count has fallen since it was queued.
  # 3. Merges it in the chunks noted for it, adjusting the counts of the pairs
  #    that each merge removes or creates, and queues the created ones.
  def train(count : Int32) : Array(Pair)
    words = [] of Array(Int32)
    weights = [] of Int64
    @chunks.each do |chunk, occurrences|
      words << chunk.bytes.map(&.to_i32)
      weights << occurrences.to_i64
    end
    pair_counts = Hash(Pair, Int64).new(0_i64)
    where = Hash(Pair, Array(Int32)).new { |hash, key| hash[key] = [] of Int32 }
    words.each_with_index do |ids, word|
      (ids.size - 1).times do |i|
        pair = {ids[i], ids[i + 1]}
        pair_counts[pair] += weights[word]
        positions = where[pair]
        positions << word unless positions.last? == word
      end
    end
    queue = PairQueue.new
    pair_counts.each { |pair, total| queue.push(total, pair) if total > 0 }

    merges = [] of Pair
    while merges.size < count
      top = queue.pop? || break
      queued, pair = top
      current = pair_counts[pair]? || 0_i64
      next if current <= 0
      if queued != current
        queue.push(current, pair)
        next
      end
      new_id = 256 + merges.size
      merges << pair
      created = Hash(Pair, Array(Int32)).new { |hash, key| hash[key] = [] of Int32 }
      where.delete(pair).try &.each do |word|
        merge_word(words, word, pair, new_id) do |changed, delta|
          pair_counts[changed] += delta * weights[word]
          if delta > 0
            positions = created[changed]
            positions << word unless positions.last? == word
          end
        end
      end
      created.each do |changed, positions|
        where[changed].concat(positions)
        n = pair_counts[changed]
        queue.push(n, changed) if n > 0
      end
    end
    merges
  end

  # Replaces each occurrence of *pair* in word *w* with *new_id*, left to
  # right without overlap, and yields each pair count change: -1 for a pair
  # removed, +1 for one created.
  private def merge_word(words : Array(Array(Int32)), w : Int32, pair : Pair, new_id : Int32, & : Pair, Int64 ->) : Nil
    ids = words[w]
    return if ids.size < 2
    a, b = pair
    result = Array(Int32).new(ids.size)
    i = 0
    while i < ids.size
      if i + 1 < ids.size && ids[i] == a && ids[i + 1] == b
        if left = result.last?
          yield({left, a}, -1_i64)
          yield({left, new_id}, 1_i64)
        end
        yield(pair, -1_i64)
        if i + 2 < ids.size
          right = ids[i + 2]
          yield({b, right}, -1_i64)
          yield({new_id, right}, 1_i64)
        end
        result << new_id
        i += 2
      else
        result << ids[i]
        i += 1
      end
    end
    words[w] = result
  end

  # A binary max-heap of {count, pair}: the highest count first, and on equal
  # counts the smaller pair.
  class PairQueue
    @items = [] of {Int64, Pair}

    def push(count : Int64, pair : Pair) : Nil
      @items << {count, pair}
      i = @items.size - 1
      while i > 0
        parent = (i - 1) // 2
        break unless before?(@items[i], @items[parent])
        @items.swap(i, parent)
        i = parent
      end
    end

    def pop? : {Int64, Pair}?
      return if @items.empty?
      top = @items.first
      last = @items.pop
      unless @items.empty?
        @items[0] = last
        i = 0
        loop do
          left, right = 2 * i + 1, 2 * i + 2
          best = i
          best = left if left < @items.size && before?(@items[left], @items[best])
          best = right if right < @items.size && before?(@items[right], @items[best])
          break if best == i
          @items.swap(i, best)
          i = best
        end
      end
      top
    end

    private def before?(x : {Int64, Pair}, y : {Int64, Pair}) : Bool
      x[0] != y[0] ? x[0] > y[0] : x[1] < y[1]
    end
  end
end
