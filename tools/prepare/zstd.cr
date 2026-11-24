# Decompression of Zstandard frames, through the system's libzstd.
module Zstd
  @[Link("zstd", pkg_config: "libzstd")]
  lib LibZstd
    fun ZSTD_decompress(dst : Void*, dst_capacity : LibC::SizeT, src : Void*, src_size : LibC::SizeT) : LibC::SizeT
    fun ZSTD_isError(code : LibC::SizeT) : LibC::UInt
    fun ZSTD_getErrorName(code : LibC::SizeT) : LibC::Char*
  end

  class Error < Exception
  end

  # Returns *src* decompressed, which must come to exactly *size* bytes.
  # Raises `Zstd::Error` for corrupt input or a different size.
  def self.decompress(src : Bytes, size : Int32) : Bytes
    dst = Bytes.new(size)
    written = LibZstd.ZSTD_decompress(dst.to_unsafe, size, src.to_unsafe, src.size)
    raise Error.new("zstd: #{String.new(LibZstd.ZSTD_getErrorName(written))}") if LibZstd.ZSTD_isError(written) != 0
    raise Error.new("zstd: #{written} bytes, expected #{size}") unless written == size
    dst
  end
end
