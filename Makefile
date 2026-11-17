# Orbix development entry points.
#
# CGO_ENABLED=0 is set per target rather than inherited from the environment,
# because a build that accidentally picks up CGo is the one failure mode this
# project cannot tolerate. Setting it here means the Makefile stays correct
# even if the caller exported CGO_ENABLED=1.

BIN      := bin/orbix
PKG      := ./...
VERSION  ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
COMMIT   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo none)
DATE     ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
LDFLAGS  := -s -w \
	-X main.version=$(VERSION) \
	-X main.commit=$(COMMIT) \
	-X main.date=$(DATE)

GOFLAGS  :=
export CGO_ENABLED := 0

# CROSS_TARGETS is every platform the project claims to build for. The list is short
# on purpose: each entry is a platform somebody runs the binary on, and a platform with
# no user is a build that breaks without anyone noticing.
CROSS_TARGETS := linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64

.PHONY: all build test test-race lint fmt vet bench clean check cross

all: check

## build: compile the orbix binary into bin/
build:
	@mkdir -p $(dir $(BIN))
	go build -trimpath -ldflags '$(LDFLAGS)' -o $(BIN) ./cmd/orbix
	@echo "built $(BIN) ($(VERSION))"

## test: run the unit test suite with CGo disabled
test:
	go test $(PKG)

## test-race: run the unit test suite under the race detector
test-race:
	go test -race $(PKG)

## vet: run go vet
vet:
	go vet $(PKG)

## fmt: format every Go file
fmt:
	gofmt -l -w .

## lint: run golangci-lint
lint:
	golangci-lint run

## bench: run kernel and engine benchmarks
bench:
	go test -run '^$$' -bench . -benchmem $(PKG)

## check: the full local gate - format, vet, test, lint
check: vet test lint
	@unformatted="$$(gofmt -l .)"; \
	if [ -n "$$unformatted" ]; then \
		echo "gofmt needed on:"; echo "$$unformatted"; exit 1; \
	fi
	@echo "check: ok"

## cross: build every supported platform with CGo disabled
cross:
	@set -e; for target in $(CROSS_TARGETS); do \
		os=$${target%/*}; arch=$${target#*/}; ext=""; \
		if [ "$$os" = "windows" ]; then ext=".exe"; fi; \
		out="bin/orbix-$$os-$$arch$$ext"; \
		echo "building $$out"; \
		GOOS=$$os GOARCH=$$arch go build -trimpath -ldflags '$(LDFLAGS)' -o "$$out" ./cmd/orbix; \
	done


## clean: remove build output
clean:
	rm -rf bin

## help: list available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
