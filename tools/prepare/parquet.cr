require "./thrift"
require "./zstd"

# Reads one string column of a Parquet file, row group by row group: the
# layout of the climbmix shards, and nothing more general.
#
# Supported: uncompressed or ZSTD column chunks of v1 data pages, PLAIN
# `BYTE_ARRAY` values, and definition levels with no nulls. Anything else
# raises `Parquet::Error` naming what it met, so a shard in another layout
# fails loudly rather than reading wrongly.
#
# ```
# file = Parquet::File.new("shard_00000.parquet")
# file.row_groups               # => 84
# file.strings(0, "text").first # => the first document
# ```
module Parquet
  class Error < Exception
  end

  MAGIC = "PAR1".to_slice

  # Compression codecs, as numbered in parquet.thrift.
  CODEC_UNCOMPRESSED = 0
  CODEC_ZSTD         = 6

  class File
    getter path : String
    getter num_rows : Int64
    # Rows in each row group, in file order.
    getter row_group_rows : Array(Int64)
    @row_groups : Array(Thrift::Struct)
    @columns : Array({String, Bool})
    @data : Bytes

    def initialize(@path : String)
      @data = ::File.open(path, &.getb_to_end)
      check_magic
      footer_size = IO::ByteFormat::LittleEndian.decode(UInt32, @data[@data.size - 8, 4]).to_i32
      metadata = Thrift::CompactReader.new(@data[@data.size - 8 - footer_size, footer_size]).read_struct
      @num_rows = Thrift.int(metadata, 3)
      @columns = leaf_columns(Thrift.list(metadata, 2))
      @row_groups = Thrift.list(metadata, 4).map(&.as(Thrift::Struct))
      @row_group_rows = @row_groups.map { |group| Thrift.int(group, 3) }
    end

    # Returns the number of row groups.
    def row_groups : Int32
      @row_groups.size
    end

    # Returns the values of string column *name* in row group *index*, in order.
    def strings(index : Int32, name : String) : Array(String)
      column = @columns.index { |(column_name, _)| column_name == name } || raise Error.new("#{path}: no column #{name}")
      optional = @columns[column][1]
      chunk = Thrift.struct(Thrift.list(@row_groups[index], 1)[column].as(Thrift::Struct), 3)
      codec = Thrift.int(chunk, 4)
      unless codec.in?(CODEC_UNCOMPRESSED, CODEC_ZSTD)
        raise Error.new("#{path}: column #{name} uses codec #{codec}; only uncompressed and ZSTD are supported")
      end
      raise Error.new("#{path}: column #{name} has a dictionary page; only PLAIN is supported") if chunk.has_key?(11_i16)
      expected = Thrift.int(chunk, 5)
      pos = Thrift.int(chunk, 9).to_i32
      values = [] of String
      while values.size < expected
        pos = read_page(pos, codec, optional, values)
      end
      values
    end

    # Reads the data page at *pos* into *values* and returns the position after it.
    private def read_page(pos : Int32, codec : Int64, optional : Bool, values : Array(String)) : Int32
      reader = Thrift::CompactReader.new(@data, pos)
      header = reader.read_struct
      type = Thrift.int(header, 1)
      raise Error.new("#{path}: page type #{type} at byte #{pos}; only v1 data pages (0) are supported") unless type == 0
      uncompressed_size = Thrift.int(header, 2).to_i32
      compressed_size = Thrift.int(header, 3).to_i32
      page = Thrift.struct(header, 5)
      count = Thrift.int(page, 1).to_i32
      raise Error.new("#{path}: encoding #{Thrift.int(page, 2)}; only PLAIN (0) is supported") unless Thrift.int(page, 2) == 0
      body = @data[reader.pos, compressed_size]
      body = Zstd.decompress(body, uncompressed_size) if codec == CODEC_ZSTD
      offset = 0
      if optional
        levels_size = IO::ByteFormat::LittleEndian.decode(UInt32, body[0, 4]).to_i32
        present = count_present(body[4, levels_size], count)
        raise Error.new("#{path}: #{count - present} null values; nulls are not supported") unless present == count
        offset = 4 + levels_size
      end
      count.times do
        size = IO::ByteFormat::LittleEndian.decode(UInt32, body[offset, 4]).to_i32
        values << String.new(body[offset + 4, size])
        offset += 4 + size
      end
      reader.pos + compressed_size
    end

    # Returns how many of *count* definition levels, RLE/bit-packed with bit
    # width 1, are 1 (a value is present).
    private def count_present(levels : Bytes, count : Int32) : Int32
      io = IO::Memory.new(levels)
      seen = present = 0
      while seen < count
        header = read_uleb(io)
        if header & 1 == 0
          run = (header >> 1).to_i32
          value = io.read_byte || raise Error.new("#{path}: truncated definition levels")
          present += run if value == 1
          seen += run
        else
          groups = (header >> 1).to_i32
          groups.times do
            bits = io.read_byte || raise Error.new("#{path}: truncated definition levels")
            8.times do |bit|
              break if seen >= count
              present += 1 if (bits >> bit) & 1 == 1
              seen += 1
            end
          end
        end
      end
      present
    end

    private def read_uleb(io : IO) : UInt64
      result = 0_u64
      shift = 0
      loop do
        b = io.read_byte || raise Error.new("#{path}: truncated varint")
        result |= (b & 0x7f).to_u64 << shift
        return result if b < 0x80
        shift += 7
      end
    end

    # Returns each leaf column's name and whether it is optional, from the
    # flattened schema; nested columns raise.
    private def leaf_columns(schema : Array(Thrift::Value)) : Array({String, Bool})
      elements = schema.map(&.as(Thrift::Struct))
      root, leaves = elements.first, elements[1..]
      raise Error.new("#{path}: nested columns are not supported") if leaves.any?(&.has_key?(5_i16)) || Thrift.int(root, 5) != leaves.size
      leaves.map do |element|
        name = String.new(element[4_i16]?.as?(Bytes) || raise Error.new("#{path}: unnamed column"))
        {name, element[3_i16]?.as?(Int64) == 1}
      end
    end

    private def check_magic : Nil
      unless @data.size >= 12 && @data[0, 4] == MAGIC && @data[@data.size - 4, 4] == MAGIC
        raise Error.new("#{path}: not a Parquet file")
      end
    end
  end
end
