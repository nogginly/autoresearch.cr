require "./spec_helper"
require "./support/data_fixtures"

describe "M3 exit: the data matches prepare.py's" do
  it "reads every Parquet shard's row groups" do
    data_meta["row_groups"].as_h.each do |name, sizes|
      Parquet::File.new("#{DATA_FIXTURES}/parquet/#{name}").row_group_rows.should eq sizes.as_a.map(&.as_i64)
    end
  end

  it "trains the tokens rustbpe trains, in the same order" do
    ours = TokenizerFiles.token_bytes(fixture_merges).map(&.hexstring)
    ours.should eq data_meta["ranks"].as_a.map(&.as_s)
  end

  it "counts each token's bytes as prepare.py does" do
    TokenizerFiles.byte_lengths(fixture_merges, Autoresearch::SPECIAL_TOKENS).should eq data_meta["token_bytes"].as_a.map(&.as_i)
  end

  it "encodes every document to prepare.py's ids, BOS first" do
    with_prepared do |tokens_dir, tokenizer|
      tokenizer.vocab_size.should eq data_meta["vocab_size"].as_i
      tokenizer.bos_token_id.should eq data_meta["bos"].as_i
      expected = data_meta["document_ids_sha256"].as_a.map(&.as_s)
      ours = data_meta["files"].as_a.flat_map do |name|
        shard = Autoresearch::TokenShard.new(File.join(tokens_dir, name.as_s.sub(".parquet", ".tokens")))
        (0...shard.documents).map { |i| shard.document(i).to_a.map(&.to_i32) }
      end
      ours.first(5).should eq data_meta["first_document_ids"].as_a.map(&.as_a.map(&.as_i))
      ours.map { |ids| int32_sha(ids) }.should eq expected
    end
  end

  it "packs the train and val batches make_dataloader packs, with its epochs" do
    batches = SafeTensors.open("#{DATA_FIXTURES}/batches.safetensors")
    with_prepared do |tokens_dir, _|
      {"train", "val"}.each do |split|
        loader_meta = data_meta["loaders"][split]
        b, t = loader_meta["batch"].as_i, loader_meta["sequence"].as_i
        loader = Autoresearch::Dataloader.new(Autoresearch::Dataloader.split(split, tokens_dir), b, t)
        loader_meta["epochs"].as_a.each_with_index do |epoch, k|
          inputs, targets, ours = loader.next_batch
          {inputs, targets, ours}.should eq({batches["#{split}.#{k}.inputs"].to_a(Int32), batches["#{split}.#{k}.targets"].to_a(Int32), epoch.as_i})
        end
      end
    end
  end
end

describe Prepare::TrainTokenizer do
  it "cuts documents to 10,000 characters and stops after the one that crosses the character limit" do
    documents = [] of String
    Prepare::TrainTokenizer.each_document("#{DATA_FIXTURES}/parquet", max_chars: 30_000) { |document| documents << document }
    documents.max_of(&.size).should eq 10_000
    documents.sum(&.size).should be >= 30_000
    (documents.sum(&.size) - documents.last.size).should be < 30_000
  end
end

describe TokenizerFiles do
  it "counts a cut UTF-8 sequence as one replacement character" do
    # "é" is C3 A9: whole, 2 bytes; its lead byte alone decodes to U+FFFD, 3 bytes.
    TokenizerFiles.replaced_size(Bytes[0xC3, 0xA9]).should eq 2
    TokenizerFiles.replaced_size(Bytes[0xC3]).should eq 3
    # "你" is E4 BD A0; its first two bytes are one maximal subpart, then "a": 3 + 1.
    TokenizerFiles.replaced_size(Bytes[0xE4, 0xBD, 0x61]).should eq 4
    # 0x80 cannot start a sequence: each is its own U+FFFD.
    TokenizerFiles.replaced_size(Bytes[0x80, 0x80]).should eq 6
  end
end

describe Autoresearch::TokenShard do
  it "round-trips documents and row groups" do
    path = File.tempname("shard", ".tokens")
    begin
      Autoresearch::TokenShard.write(path, [2, 1], [[8188, 1, 2], [8188], [8188, 65535]])
      shard = Autoresearch::TokenShard.new(path)
      shard.row_group_sizes.should eq [2, 1]
      shard.documents.should eq 3
      (0...3).map { |i| shard.document(i).to_a }.should eq [[8188u16, 1u16, 2u16], [8188u16], [8188u16, 65535u16]]
    ensure
      File.delete?(path)
    end
  end
end
