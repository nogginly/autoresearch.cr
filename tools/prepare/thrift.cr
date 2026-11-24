# A reader for Thrift's compact protocol, as Parquet stores its metadata.
#
# Structs are read whole into `Thrift::Struct`s, field id to value, so a
# caller picks out the fields it knows and every other field is skipped.
module Thrift
  alias Value = Int64 | Float64 | Bool | Bytes | Array(Value) | Struct
  alias Struct = Hash(Int16, Value)

  class Error < Exception
  end

  # Reads compact-protocol values from *bytes*, starting at *pos*.
  class CompactReader
    getter pos : Int32

    def initialize(@bytes : Bytes, @pos : Int32 = 0)
    end

    # Returns the struct that starts at the current position, and moves past it.
    def read_struct : Struct
      result = Struct.new
      last = 0_i16
      loop do
        header = byte
        return result if header == 0
        delta = (header >> 4).to_i16
        type = header & 0x0f
        id = delta == 0 ? zigzag(varint).to_i16 : last + delta
        last = id
        result[id] = case type
                     when 1 then true
                     when 2 then false
                     else        read_value(type)
                     end
      end
    end

    private def read_value(type : UInt8) : Value
      case type
      when 3       then byte.to_i8.to_i64
      when 4, 5, 6 then zigzag(varint)
      when 7
        value = IO::ByteFormat::LittleEndian.decode(Float64, @bytes[@pos, 8])
        @pos += 8
        value
      when 8
        size = varint.to_i32
        value = @bytes[@pos, size]
        @pos += size
        value
      when 9, 10 then read_list
      when 12    then read_struct
      else
        raise Error.new("thrift: unsupported compact type #{type} at byte #{@pos}")
      end
    end

    private def read_list : Value
      header = byte
      size = (header >> 4).to_i32
      size = varint.to_i32 if size == 15
      type = header & 0x0f
      Array(Value).new(size) { type == 1 || type == 2 ? byte == 1 : read_value(type) }
    end

    private def byte : UInt8
      raise Error.new("thrift: unexpected end at byte #{@pos}") if @pos >= @bytes.size
      value = @bytes[@pos]
      @pos += 1
      value
    end

    private def varint : UInt64
      result = 0_u64
      shift = 0
      loop do
        b = byte
        result |= (b & 0x7f).to_u64 << shift
        return result if b < 0x80
        shift += 7
      end
    end

    private def zigzag(n : UInt64) : Int64
      (n >> 1).to_i64 ^ -(n & 1).to_i64
    end
  end

  # Returns field *id* of *s* as an integer, raising if it is missing.
  def self.int(s : Struct, id : Int) : Int64
    s[id.to_i16]?.as?(Int64) || raise Error.new("thrift: field #{id} is not an integer")
  end

  # Returns field *id* of *s* as a struct, raising if it is missing.
  def self.struct(s : Struct, id : Int) : Struct
    s[id.to_i16]?.as?(Struct) || raise Error.new("thrift: field #{id} is not a struct")
  end

  # Returns field *id* of *s* as a list, raising if it is missing.
  def self.list(s : Struct, id : Int) : Array(Value)
    s[id.to_i16]?.as?(Array(Value)) || raise Error.new("thrift: field #{id} is not a list")
  end
end
