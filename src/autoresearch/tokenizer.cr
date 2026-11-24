require "json"
require "tensomite"
require "./constants"

module Autoresearch
  # The trained tokenizer, as `prepare train-tokenizer` saves it, and the byte
  # length of each token for the bits-per-byte metric.
  class Tokenizer
    getter bpe : Tensomite::BPE
    getter bos_token_id : Int32
    # The UTF-8 length of each token, 0 for special tokens.
    getter token_bytes : Array(Int32)

    def initialize(@bpe : Tensomite::BPE, @token_bytes : Array(Int32))
      @bos_token_id = bpe.id?(BOS_TOKEN) || raise Tensomite::Error.new("the tokenizer has no #{BOS_TOKEN}")
    end

    # Loads `tokenizer.json` and `token_bytes.json` from *dir*.
    def self.from_directory(dir : String = Autoresearch.tokenizer_dir) : Tokenizer
      new(Tensomite::BPE.from_json(File.join(dir, "tokenizer.json")),
        Array(Int32).from_json(File.read(File.join(dir, "token_bytes.json"))))
    end

    def vocab_size : Int32
      bpe.size
    end

    # Returns *text*'s token ids, special tokens in it encoded as text, led by
    # the BOS token when *bos*.
    def encode(text : String, bos : Bool = false) : Array(Int32)
      ids = bpe.encode(text, special: false)
      ids.unshift(bos_token_id) if bos
      ids
    end

    def decode(ids : Enumerable(Int32)) : String
      bpe.decode(ids)
    end
  end
end
