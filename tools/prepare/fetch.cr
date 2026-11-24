require "http/client"
require "../../src/autoresearch/constants"

module Prepare
  # Downloads Parquet shards into the data directory, as prepare.py's
  # `download_data` does: the first *num_shards* training shards and the
  # pinned validation shard, *workers* at a time. A shard already present is
  # skipped; each is written to `.tmp` and renamed when complete, and retried
  # up to 5 times with backoff.
  module Fetch
    MAX_ATTEMPTS  = 5
    MAX_REDIRECTS = 5

    # Returns the shard indices to download for *num_shards* training shards.
    def self.indices(num_shards : Int32) : Array(Int32)
      ids = (0...Math.min(num_shards, Autoresearch::MAX_SHARD)).to_a
      ids << Autoresearch::VAL_SHARD unless ids.includes?(Autoresearch::VAL_SHARD)
      ids
    end

    def self.run(num_shards : Int32, workers : Int32, dir : String = Autoresearch.data_dir) : Nil
      Dir.mkdir_p(dir)
      ids = indices(num_shards)
      missing = ids.reject { |i| File.exists?(File.join(dir, Autoresearch.shard_name(i, "parquet"))) }
      if missing.empty?
        puts "Data: all #{ids.size} shards already downloaded at #{dir}"
        return
      end
      puts "Data: downloading #{missing.size} shards (#{ids.size - missing.size} already exist)..."
      queue = Channel(Int32).new(missing.size)
      missing.each { |i| queue.send(i) }
      queue.close
      done = Channel(Bool).new
      worker_count = Math.max(1, Math.min(workers, missing.size))
      worker_count.times do
        spawn do
          while index = queue.receive?
            done.send(download(index, dir))
          end
        end
      end
      ok = missing.count { done.receive }
      puts "Data: #{ids.size - missing.size + ok}/#{ids.size} shards ready at #{dir}"
      abort "Data: #{missing.size - ok} shards failed to download" if ok < missing.size
    end

    # Downloads shard *index*, returning whether it succeeded.
    def self.download(index : Int32, dir : String) : Bool
      name = Autoresearch.shard_name(index, "parquet")
      path = File.join(dir, name)
      temporary = "#{path}.tmp"
      (1..MAX_ATTEMPTS).each do |attempt|
        get("#{Autoresearch::BASE_URL}/#{name}", temporary)
        File.rename(temporary, path)
        puts "  Downloaded #{name}"
        return true
      rescue error
        puts "  Attempt #{attempt}/#{MAX_ATTEMPTS} failed for #{name}: #{error.message}"
        File.delete?(temporary)
        sleep (2 ** attempt).seconds if attempt < MAX_ATTEMPTS
      end
      false
    end

    # Streams *url* to *path*, following redirects. Raises for any status but 200.
    private def self.get(url : String, path : String) : Nil
      MAX_REDIRECTS.times do
        uri = URI.parse(url)
        location = nil
        HTTP::Client.new(uri) do |client|
          client.connect_timeout = 30.seconds
          client.read_timeout = 60.seconds
          client.get(uri.request_target) do |response|
            if response.status.redirection?
              location = response.headers["Location"]? || raise IO::Error.new("redirect without a location")
            elsif response.status.success?
              File.open(path, "w") { |file| IO.copy(response.body_io, file) }
              return
            else
              raise IO::Error.new("HTTP #{response.status_code}")
            end
          end
        end
        url = uri.resolve(location || raise IO::Error.new("redirect without a location")).to_s
      end
      raise IO::Error.new("more than #{MAX_REDIRECTS} redirects")
    end
  end
end
