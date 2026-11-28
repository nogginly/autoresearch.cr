"""Checks bin/prepare's output on real data against prepare.py's own, at the pinned commit.

Usage: python check_prepared.py AUTORESEARCH_DIR CACHE_DIR BATCH_DIGESTS_JSON

CACHE_DIR is the cache bin/prepare filled (its data/, tokenizer/ and tokens/).
prepare.py trains its own tokenizer on the same Parquet shards into
CACHE_DIR/reference/, then the script compares:

1. every token's bytes and byte length with tokenizer/;
2. every document's ids, BOS first, in every shard, with tokens/;
3. the first batches of make_dataloader (patched to stay on the CPU) with
   BATCH_DIGESTS_JSON, which bin/batch_digests writes, packed at the batch
   shape the file records.

Prints one line per check and exits non-zero on any difference.
"""

import hashlib
import json
import os
import struct
import sys

COMMIT = "228791fb499afffb54b46200aca536f79142f117"
PREPARE_SHA256 = "4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150"
PATCHES = [
    ("pin_memory=True", "pin_memory=False"),
    ('gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")',
     'gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cpu")'),
]


def load_prepare(source_dir, cache_dir):
    path = os.path.join(source_dir, "prepare.py")
    code = open(path, "rb").read()
    if hashlib.sha256(code).hexdigest() != PREPARE_SHA256:
        sys.exit(f"{path} is not prepare.py at {COMMIT}")
    code = code.decode()
    for old, new in PATCHES:
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    prepare = {"__name__": "autoresearch_prepare", "__file__": path}
    exec(compile(code, path, "exec"), prepare)
    prepare["DATA_DIR"] = os.path.join(cache_dir, "data")
    prepare["TOKENIZER_DIR"] = os.path.join(cache_dir, "reference")
    return prepare


def byte_level_decoder():
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars, n = printable[:], 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(printable, chars)}


def read_token_shard(path):
    """Yields (row group sizes, documents) of a token shard, documents as lists of ids."""
    data = open(path, "rb").read()
    assert data[:8] == b"ARTOKEN1", path
    groups, _, docs, tokens = struct.unpack_from("<IIQQ", data, 8)
    sizes = list(struct.unpack_from(f"<{groups}I", data, 32))
    at = 32 + (4 * groups + 7) // 8 * 8
    offsets = struct.unpack_from(f"<{docs + 1}Q", data, at)
    base = at + 8 * (docs + 1)
    ids = memoryview(data)[base:base + 2 * tokens].cast("H")
    return sizes, [ids[offsets[i]:offsets[i + 1]] for i in range(docs)]


def main():
    source_dir, cache_dir, digests_path = sys.argv[1:4]
    prepare = load_prepare(source_dir, cache_dir)
    failures = 0

    # 1. The tokenizer.
    prepare["train_tokenizer"]()
    tokenizer = prepare["Tokenizer"].from_directory(prepare["TOKENIZER_DIR"])
    enc = tokenizer.enc
    theirs = [token for token, _ in sorted(enc._mergeable_ranks.items(), key=lambda item: item[1])]
    ours_json = json.load(open(os.path.join(cache_dir, "tokenizer", "tokenizer.json")))
    decode = byte_level_decoder()
    ours = [None] * len(ours_json["model"]["vocab"])
    for text, i in ours_json["model"]["vocab"].items():
        ours[i] = bytes(decode[c] for c in text)
    same = ours == theirs
    failures += not same
    print(f"tokens: {len(theirs)} ranked, ours {len(ours)}; identical: {same}")
    bytes_same = json.load(open(os.path.join(cache_dir, "tokenizer", "token_bytes.json"))) == prepare["get_token_bytes"]().tolist()
    failures += not bytes_same
    print(f"token byte lengths: identical: {bytes_same}")

    # 2. Every document of every shard, row group by row group.
    import pyarrow.parquet as pq
    bos = tokenizer.get_bos_token_id()
    total_docs = total_tokens = differing = 0
    for name in sorted(f for f in os.listdir(prepare["DATA_DIR"]) if f.endswith(".parquet")):
        sizes, docs = read_token_shard(os.path.join(cache_dir, "tokens", name.replace(".parquet", ".tokens")))
        pf = pq.ParquetFile(os.path.join(prepare["DATA_DIR"], name))
        assert sizes == [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)], name
        at = 0
        for g in range(pf.num_row_groups):
            texts = pf.read_row_group(g).column("text").to_pylist()
            for ids in tokenizer.encode(texts, prepend=bos):
                if list(docs[at]) != ids:
                    differing += 1
                total_tokens += len(ids)
                at += 1
        total_docs += at
    failures += differing > 0
    print(f"documents: {total_docs} in all shards, {total_tokens} tokens; differing: {differing}")

    # 3. The first batches of each split.
    ours = json.load(open(digests_path))
    b, t = ours["batch"], ours["sequence"]
    for split, batches in ours["splits"].items():
        loader = prepare["make_dataloader"](tokenizer, b, t, split)
        mismatched = 0
        for digest, epoch in batches:
            x, y, their_epoch = next(loader)
            flat = x.reshape(-1).tolist() + y.reshape(-1).tolist()
            theirs = hashlib.sha256(struct.pack(f"<{len(flat)}i", *flat)).hexdigest()
            mismatched += theirs != digest or their_epoch != epoch
        failures += mismatched > 0
        print(f"{split} batches: {len(batches)} of {b} × {t}; differing: {mismatched}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
