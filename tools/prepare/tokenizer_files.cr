require "json"
require "tensomite"

# The files a trained tokenizer is saved as: a Hugging Face `tokenizer.json`
# that `Tensomite::BPE.from_json` reads, and the byte length of each token for
# the bits-per-byte metric.
module TokenizerFiles
  # Returns each token's bytes, ids 0 to 255 the single bytes, then one per
  # merge, in id order.
  def self.token_bytes(merges : Array({Int32, Int32})) : Array(Bytes)
    tokens = (0..255).map { |byte| Bytes[byte.to_u8] }
    merges.each do |(left, right)|
      joined = Bytes.new(tokens[left].size + tokens[right].size)
      tokens[left].copy_to(joined)
      tokens[right].copy_to(joined + tokens[left].size)
      tokens << joined
    end
    tokens
  end

  # Writes a `tokenizer.json` for *merges* and *specials*, which take the ids
  # after the merged tokens in order, splitting text with *pattern*: a `Split`
  # pre-tokeniser before a `ByteLevel` with its own pattern off, as
  # tensomite's `scripts/tiktoken_to_json.py` writes for a tiktoken encoding.
  def self.write_json(path : String, merges : Array({Int32, Int32}), specials : Array(String), pattern : String) : Nil
    tokens = token_bytes(merges)
    text = ->(bytes : Bytes) { String.build { |io| bytes.each { |byte| io << Tensomite::BPE::BYTE_TO_CHAR[byte] } } }
    File.open(path, "w") do |file|
      JSON.build(file, indent: 1) do |json|
        json.object do
          json.field "version", "1.0"
          json.field "added_tokens" do
            json.array do
              specials.each_with_index do |special, i|
                json.object do
                  json.field "id", tokens.size + i
                  json.field "content", special
                  {"single_word", "lstrip", "rstrip", "normalized"}.each { |key| json.field key, false }
                  json.field "special", true
                end
              end
            end
          end
          json.field "normalizer", nil
          json.field "pre_tokenizer" do
            json.object do
              json.field "type", "Sequence"
              json.field "pretokenizers" do
                json.array do
                  json.object do
                    json.field "type", "Split"
                    json.field "pattern" { json.object { json.field "Regex", pattern } }
                    json.field "behavior", "Isolated"
                    json.field "invert", false
                  end
                  json.object do
                    json.field "type", "ByteLevel"
                    json.field "add_prefix_space", false
                    json.field "trim_offsets", true
                    json.field "use_regex", false
                  end
                end
              end
            end
          end
          json.field "decoder" do
            json.object do
              json.field "type", "ByteLevel"
              json.field "add_prefix_space", true
              json.field "trim_offsets", true
              json.field "use_regex", true
            end
          end
          json.field "model" do
            json.object do
              json.field "type", "BPE"
              {"dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix"}.each { |key| json.field key, nil }
              json.field "fuse_unk", false
              json.field "byte_fallback", false
              json.field "vocab" do
                json.object { tokens.each_with_index { |token, id| json.field text.call(token), id } }
              end
              json.field "merges" do
                json.array { merges.each { |(left, right)| json.string "#{text.call(tokens[left])} #{text.call(tokens[right])}" } }
              end
            end
          end
        end
      end
    end
  end

  # Returns the byte length of each token, as prepare.py computes it: the
  # UTF-8 length of the token decoded with errors replaced (see
  # `.replaced_size`), and 0 for the *specials* after it.
  def self.byte_lengths(merges : Array({Int32, Int32}), specials : Array(String)) : Array(Int32)
    token_bytes(merges).map { |bytes| replaced_size(bytes) } + Array.new(specials.size, 0)
  end

  # Returns the UTF-8 length of *bytes* decoded as Python's
  # `bytes.decode("utf-8", errors="replace")` decodes it: each maximal invalid
  # subpart becomes U+FFFD, three bytes.
  #
  # 1. A valid sequence of 1 to 4 bytes keeps its length.
  # 2. A lead byte followed by fewer valid continuation bytes than it needs is
  #    one subpart, as is any byte that cannot start a sequence.
  def self.replaced_size(bytes : Bytes) : Int32
    size = i = 0
    while i < bytes.size
      lead = bytes[i]
      need, low, high = sequence(lead)
      if need == 0
        size += lead < 0x80 ? 1 : 3
        i += 1
        next
      end
      taken = 1
      while taken <= need && i + taken < bytes.size
        byte = bytes[i + taken]
        ok = taken == 1 ? low <= byte <= high : 0x80 <= byte <= 0xBF
        break unless ok
        taken += 1
      end
      size += taken == need + 1 ? need + 1 : 3
      i += taken
    end
    size
  end

  # Returns how many continuation bytes *lead* needs, and the range its first
  # continuation byte must fall in; 0 continuation bytes for ASCII and for
  # bytes that cannot start a sequence.
  private def self.sequence(lead : UInt8) : {Int32, UInt8, UInt8}
    case lead
    when 0xC2..0xDF then {1, 0x80_u8, 0xBF_u8}
    when 0xE0       then {2, 0xA0_u8, 0xBF_u8}
    when 0xE1..0xEC then {2, 0x80_u8, 0xBF_u8}
    when 0xED       then {2, 0x80_u8, 0x9F_u8}
    when 0xEE..0xEF then {2, 0x80_u8, 0xBF_u8}
    when 0xF0       then {3, 0x90_u8, 0xBF_u8}
    when 0xF1..0xF3 then {3, 0x80_u8, 0xBF_u8}
    when 0xF4       then {3, 0x80_u8, 0x8F_u8}
    else                 {0, 0_u8, 0_u8}
    end
  end
end
