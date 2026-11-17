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

.PHONY: all build test test-race lint fmt vet bench clean check cross docs

all: check

## build: compile the orbix binary into bin/
build:
	@mkdir -p $(dir $(BIN))
	go build -trimpath -ldflags '$(LDFLAGS)' -o $(BIN) ./cmd/orbix
	@echo "built $(BIN) ($(VERSION))"

## test: run the unit test suite with CGo disabled
##
## -count=1 disables the test result cache. A cached pass is a pass that did not run,
## and the gate exists to run things: a contributor who changes a test helper and sees
## green from a cached binary has learned nothing.
test:
	go test -count=1 $(PKG)

## test-race: run the unit test suite under the race detector
##
## The race detector requires cgo, so this target overrides the CGo ban for itself only.
## That does not weaken the build: the ban exists so no shipping binary ever links C, and
## this target produces no binary -- it is a test instrument. Without the override the
## target could never have succeeded, because the Makefile exports CGO_ENABLED=0 to
## every recipe.
test-race:
	CGO_ENABLED=1 go test -count=1 -race $(PKG)

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

## docs: lint Markdown with rumdl
##
## CI runs this as its own job, and the repository's convention is that Markdown is code:
## the README is the project's front page and each package's doc.go is its authoritative
## description, so a malformed table or a broken link is a defect like any other. It is a
## separate target rather than part of `check` because rumdl is a Python-era tool that a
## Go-only contributor may not have, and `check` should stay runnable with the Go
## toolchain and golangci-lint alone.
docs:
	@command -v rumdl >/dev/null 2>&1 || { \
		echo "rumdl not found: install it (https://rumdl.dev) or run CI's docs job"; \
		exit 1; \
	}
	rumdl check .

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
