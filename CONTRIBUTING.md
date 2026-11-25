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
just xcheck    # compare a GGUF file against the Python gguf parser, tensor by tensor
just stress    # race binary under mixed load: plain, SSE, and clients that hang up; needs MODEL=
just fixture   # write a synthetic checkpoint big enough for just stress to interrupt
just skips     # count the tests that skip for want of a checkpoint, which CI reports too
just isolate   # run the suite twice with shuffling, to catch order dependence
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

## Cross-checking a checkpoint

`just MODEL=/path/to.gguf xcheck` parses the same file with the Python `gguf` package -- an
implementation nobody here wrote -- and compares every tensor's type, dimensions and the
SHA-256 of its stored bytes. It needs uv and `just build` first.

This is the check for the assumption a repository cannot test against itself. The parser, the
fixtures and the converter that wrote them share every assumption, so a wrong one is invisible
to all three: a dimension order reversed, a block layout misread, or an offset measured from the
wrong base all produce plausible numbers rather than an error. A second reader is what notices.

## Stressing the server without a checkpoint

`just stress` needs a model to generate from and a request long enough to interrupt mid-stream.
`just fixture` writes one -- random weights, a few megabytes -- so the disconnect path is
reachable on a machine with no checkpoint:

```bash
just fixture && just MODEL=.cache/stress.gguf stress
```

The weights are random, so the generated text is meaningless. That is fine: the script measures
whether a disconnect is handled, not what the model said. It fails on a DATA RACE, a 5xx, an
active gauge that did not return to zero, or a run in which nothing was cancelled -- the last
condition is what keeps it from reporting ok after testing nothing.

Measured on four checkpoints -- 33, 57, 201 and 290 tensors -- every tensor agreed. The script's
own failure modes were checked by breaking them: shifting the reported offset by one byte fails
every tensor with a neighbour, and reversing the reported dimensions fails every non-square one.
The second is worth knowing about because it is the mistake to make while writing such a
comparison: `gguf`'s `shape` is already in the file's stored order, the same order orbix prints
as `dims`, and reversing one side to "match" turns 112 of TinyLlama's 201 tensors into reported
disagreements.

## Constraints CI enforces

CI enforces `CGO_ENABLED=0` on every platform. A build that requires CGo is a build that fails.

Renovate keeps the workflow's pinned actions current: `.github/renovate.json` extends
`config:recommended`, so a new `actions/checkout`, `actions/setup-go`, or lint action release
arrives as a pull request rather than as a stale pin nobody notices. Nothing in the build
depends on it -- the versions in `ci.yml` are the ones used.

## Contributing with an AI agent

Read [`AGENTS.md`](AGENTS.md) first. It records the no-CGo constraint, layering rules, interface
contracts, and naming conventions.
