CRYSTAL     ?= crystal
SHARDS      ?= shards
AMEBA       ?= bin/ameba
SPEC_FLAGS  ?= --order random
BUILD_FLAGS ?= --release

FORMAT_PATHS := $(wildcard train.cr src spec tools)

.PHONY: deps spec lint format build test clean

## deps: install shards and build the linter, if missing
deps: lib/.shards.info $(AMEBA)

lib/.shards.info: shard.yml shard.lock
	$(SHARDS) install
	@touch $@

$(AMEBA): | lib/.shards.info
	mkdir -p bin
	cp lib/ameba/bin/ameba.cr bin/
	$(CRYSTAL) build -o $(AMEBA) bin/ameba.cr

## spec: run every spec, in random order
spec: lib/.shards.info
	$(CRYSTAL) spec $(SPEC_FLAGS)

## lint: check formatting and run ameba
lint: deps
	$(CRYSTAL) tool format --check $(FORMAT_PATHS)
	$(AMEBA)

## format: rewrite sources in the standard format
format:
	$(CRYSTAL) tool format $(FORMAT_PATHS)

## build: build the command-line programs into bin/ (release by default)
build: bin/gpt_spike

bin/gpt_spike: spikes/gpt_spike.cr lib/.shards.info FORCE
	mkdir -p bin
	$(CRYSTAL) build $(BUILD_FLAGS) -o $@ $<

## test: run the specs and the linter
test: spec lint

## clean: remove built programs, keeping the linter
clean:
	find bin -type f ! -name 'ameba*' -delete 2>/dev/null || true

FORCE:
