CRYSTAL     ?= crystal
SHARDS      ?= shards
AMEBA       ?= bin/ameba
SPEC_FLAGS  ?= --order random
BUILD_FLAGS ?= --release
CUDA        ?=

REFERENCE   ?=

DEVICE_FLAGS := $(if $(CUDA),-Dcuda) $(if $(REFERENCE),-Dreference)

FORMAT_PATHS := $(wildcard train.cr src spec tools bench)

.PHONY: deps spec lint format build test clean fixtures check-data bench profile

## deps: install shards and build the linter, if missing
deps: lib/.shards.info $(AMEBA)

lib/.shards.info: shard.yml shard.lock
	$(SHARDS) install
	@touch $@

$(AMEBA): | lib/.shards.info
	mkdir -p bin
	cp lib/ameba/bin/ameba.cr bin/
	$(CRYSTAL) build -o $(AMEBA) bin/ameba.cr

## spec: run every spec, in random order; CUDA=1 adds the CUDA device
spec: lib/.shards.info
	$(CRYSTAL) spec -Dautoresearch_spec $(DEVICE_FLAGS) $(SPEC_FLAGS)

## lint: check formatting and run ameba
lint: deps
	$(CRYSTAL) tool format --check $(FORMAT_PATHS)
	$(AMEBA)

## format: rewrite sources in the standard format
format:
	$(CRYSTAL) tool format $(FORMAT_PATHS)

## build: build the command-line programs into bin/ (release by default)
build: bin/train bin/prepare bin/batch_digests

bin/prepare: tools/prepare.cr lib/.shards.info FORCE
	mkdir -p bin
	$(CRYSTAL) build $(BUILD_FLAGS) -o $@ $<

bin/train: train.cr lib/.shards.info FORCE
	mkdir -p bin
	$(CRYSTAL) build $(BUILD_FLAGS) $(DEVICE_FLAGS) -o $@ $<

## fixtures: regenerate spec/fixtures from the pinned original (needs Python 3 and the network)
fixtures:
	scripts/fixtures/regenerate.sh

## bench: measure the optimizer step against a training step (DEVICE=blas|metal|cuda, BATCH=sequences)
bench: lib/.shards.info
	$(CRYSTAL) run --release -Dautoresearch_spec $(DEVICE_FLAGS) bench/optimizer_step.cr -- $(or $(DEVICE),blas) $(or $(BATCH),1)

## profile: profile training steps on a GPU (DEVICE=metal|cuda, BATCH, STEPS, SEQ, DEPTH)
profile: lib/.shards.info
	$(CRYSTAL) run --release -Dautoresearch_spec $(DEVICE_FLAGS) bench/profile_step.cr -- $(DEVICE) $(or $(BATCH),8) $(or $(STEPS),1) $(or $(SEQ),-) $(or $(DEPTH),-)

## check-data: compare bin/prepare's output in the cache with prepare.py's on the same shards (needs Python)
check-data: bin/batch_digests
	scripts/fixtures/regenerate.sh check

bin/batch_digests: tools/batch_digests.cr lib/.shards.info FORCE
	mkdir -p bin
	$(CRYSTAL) build $(BUILD_FLAGS) -o $@ $<

## test: run the specs and the linter
test: spec lint

## clean: remove built programs, keeping the linter
clean:
	find bin -type f ! -name 'ameba*' -delete 2>/dev/null || true

FORCE:
