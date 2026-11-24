require "tensomite"

module Autoresearch
  # A token shard: the documents of one Parquet shard, each already encoded
  # and led by the BOS token, as `prepare tokenize` writes them.
  #
  # Layout, little-endian:
  #
  # | Bytes | Contents |
  # |---|---|
  # | 8 | magic, `ARTOKEN1` |
  # | 4 | row groups, `g` |
  # | 4 | reserved, 0 |
  # | 8 | documents, `d` |
  # | 8 | tokens, `n` |
  # | 4 × g, padded to a multiple of 8 | documents in each row group of the Parquet shard |
  # | 8 × (d + 1) | each document's first token, then `n` |
  # | 2 × n | tokens, `UInt16` |
  #
  # The row groups are kept because the dataloader reads documents in
  # batches that never cross one, as prepare.py's does.
  class TokenShard
    MAGIC = "ARTOKEN1"

    getter path : String
    # Documents in each row group, in order.
    getter row_group_sizes : Array(Int32)
    getter documents : Int32
    getter tokens : Int64
    @offsets : Slice(Int64)
    @data : Slice(UInt16)

    # Opens the shard at *path*, memory-mapped. Raises `Tensomite::Error` for a file in another format.
    def initialize(@path : String)
      @file = Tensomite::MappedFile.new(path)
      header = @file.bytes(0, 32)
      raise Tensomite::Error.new("#{path}: not a token shard") unless String.new(header[0, 8]) == MAGIC
      groups = IO::ByteFormat::LittleEndian.decode(UInt32, header[8, 4]).to_i32
      @documents = IO::ByteFormat::LittleEndian.decode(UInt64, header[16, 8]).to_i32
      @tokens = IO::ByteFormat::LittleEndian.decode(UInt64, header[24, 8]).to_i64
      sizes_bytes = TokenShard.padded(4 * groups)
      sizes = @file.bytes(32, 4 * groups)
      @row_group_sizes = Array.new(groups) { |i| IO::ByteFormat::LittleEndian.decode(UInt32, sizes[4 * i, 4]).to_i32 }
      offsets_at = 32_i64 + sizes_bytes
      @offsets = @file.bytes(offsets_at, 8 * (@documents + 1)).unsafe_slice_of(Int64)
      @data = @file.bytes(offsets_at + 8 * (@documents + 1), (2 * @tokens).to_i32).unsafe_slice_of(UInt16)
    end

    # Returns document *index*'s tokens, BOS first, without copying.
    def document(index : Int32) : Slice(UInt16)
      @data[@offsets[index], @offsets[index + 1] - @offsets[index]]
    end

    # Writes a shard of *documents*, each a list of token ids, grouped as
    # *row_group_sizes* says, to *path*, by way of a temporary file renamed
    # into place when complete.
    def self.write(path : String, row_group_sizes : Array(Int32), documents : Array(Array(Int32))) : Nil
      raise ArgumentError.new("row groups hold #{row_group_sizes.sum} documents, not #{documents.size}") unless row_group_sizes.sum == documents.size
      temporary = "#{path}.tmp"
      File.open(temporary, "w") do |file|
        io = IO::ByteFormat::LittleEndian
        tokens = documents.sum(0_i64, &.size)
        file << MAGIC
        io.encode(row_group_sizes.size.to_u32, file)
        io.encode(0_u32, file)
        io.encode(documents.size.to_u64, file)
        io.encode(tokens.to_u64, file)
        row_group_sizes.each { |size| io.encode(size.to_u32, file) }
        (padded(4 * row_group_sizes.size) - 4 * row_group_sizes.size).times { file.write_byte(0) }
        offset = 0_i64
        documents.each do |document|
          io.encode(offset, file)
          offset += document.size
        end
        io.encode(offset, file)
        documents.each do |document|
          document.each do |id|
            raise ArgumentError.new("token id #{id} does not fit in UInt16") unless 0 <= id <= UInt16::MAX
            io.encode(id.to_u16, file)
          end
        end
      end
      File.rename(temporary, path)
    end

    protected def self.padded(size : Int32) : Int32
      (size + 7) // 8 * 8
    end
  end
end
