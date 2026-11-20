# Contributing to Orbix

This is primarily a personal learning project, but issues and design discussions are welcome --
especially if you are learning LLM inference yourself and spot something unclear.

## Getting started

The commands below are [just](https://just.systems) recipes. `just` is not usually preinstalled:
`cargo install just`, `brew install just`, `winget install --id Casey.Just --exact`, or a
release binary. Each recipe is a short shell command; the justfile is the readable list.

```bash
git clone https://github.com/pplmx/orbix
cd orbix

just test      # CGO_ENABLED=0 go test ./...
just lint      # golangci-lint run
just check     # the local gate: vet, test, lint, gofmt
just docs      # rumdl check: Markdown is code here too
just hooks     # prek: whitespace, final newlines, and YAML/TOML/JSON syntax
just commits   # cz check: commit subjects across a range; override RANGE=base..HEAD first
just release-preview # what the next release would be; changes nothing
just fuzz      # run every fuzz target; FUZZTIME sets the per-target budget
just build     # CGO_ENABLED=0 go build -o bin/orbix ./cmd/orbix
just bench     # run kernel and engine benchmarks
just cross     # build all five supported platforms with CGo disabled
just test-race # the suite under the race detector; needs cgo, so it is separate
```

## The gates, and where they differ

`just check` is what a contributor runs before pushing; it needs the Go toolchain and
golangci-lint **v2**, because `.golangci.yml` is a v2 configuration and a v1 binary refuses to
read one. The recipe checks the version and says so, rather than letting the linter's own
message about the config stand in for it.

`just docs` runs `rumdl check .` over the repository's Markdown, and needs rumdl. `just hooks`
runs the pre-commit hooks through prek. Both are separate targets rather than part of `check`
because neither tool is in the Go toolchain, and `check` should stay runnable with Go and
golangci-lint alone.

They differ in what they do on a problem: `just docs` reports and leaves the file alone, while
`just hooks` **rewrites** it (`rumdl` and `rumdl-fmt` pass `--fix`, and `end-of-file-fixer` and
`mixed-line-ending` edit in place). A hook that changed something exits non-zero; re-stage the
file and look at what it did rather than assuming the edit was cosmetic. `just check` sees Go
files only, which is why a stray whitespace line under `.github/workflows/` leaves it green and
`just hooks` catches it.

Run `prek install` once to make the hooks fire on every commit; it writes shims under
`.git/hooks/`, which git does not track.

## Fuzzing

`just fuzz` searches rather than gates. Six targets cover the paths that take input this
project did not produce: the GGUF container, the quantization decoders, the chat template
parser, and the tokenizer's pre-split, decode, and encode/decode round trip. Each keeps a seed
corpus that `go test` replays on every run, so a crash found once stays fixed; reading a file or
a user's text is the part that runs on someone else's input, and a panic during it takes the
process down rather than returning an error.

## Constraints CI enforces

CI enforces `CGO_ENABLED=0` on every platform. A build that requires CGo is a build that fails.

Renovate keeps the workflow's pinned actions current: `.github/renovate.json` extends
`config:recommended`, so a new `actions/checkout`, `actions/setup-go`, or lint action release
arrives as a pull request rather than as a stale pin nobody notices. Nothing in the build
depends on it -- the versions in `ci.yml` are the ones used.

## Contributing with an AI agent

Read [`AGENTS.md`](AGENTS.md) first. It records the no-CGo constraint, layering rules, interface
contracts, and naming conventions.
