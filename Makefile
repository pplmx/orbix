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

.PHONY: all build test lint fmt vet bench clean check

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
check: vet test
	@unformatted="$$(gofmt -l .)"; \
	if [ -n "$$unformatted" ]; then \
		echo "gofmt needed on:"; echo "$$unformatted"; exit 1; \
	fi
	@echo "check: ok"

## clean: remove build output
clean:
	rm -rf bin

## help: list available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
