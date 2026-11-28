# Prints the SHA-256 of each of the first batches the dataloader packs from the
# token shards in the cache, as JSON with the batch shape, for
# scripts/fixtures/check_prepared.py to compare with prepare.py's make_dataloader
# packing the same shape.
#
# Usage: bin/batch_digests [batch] [sequence] [steps]
require "digest/sha256"
require "json"
require "../src/autoresearch/dataloader"

b = (ARGV[0]? || "8").to_i
t = (ARGV[1]? || Autoresearch::MAX_SEQ_LEN.to_s).to_i
steps = (ARGV[2]? || "50").to_i

# Returns the SHA-256 of *inputs* then *targets*, as little-endian int32s.
def digest(inputs : Array(Int32), targets : Array(Int32)) : String
  io = IO::Memory.new
  (inputs + targets).each { |id| io.write_bytes(id, IO::ByteFormat::LittleEndian) }
  Digest::SHA256.hexdigest(io.to_slice)
end

splits = {"train", "val"}.to_h do |split|
  loader = Autoresearch::Dataloader.new(Autoresearch::Dataloader.split(split), b, t)
  {split, Array.new(steps) { inputs, targets, epoch = loader.next_batch; {digest(inputs, targets), epoch} }}
end
puts({"batch" => b, "sequence" => t, "splits" => splits}.to_json)
