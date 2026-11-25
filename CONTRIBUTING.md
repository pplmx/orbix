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
just jinja-diff # compare the chat template subset against real Jinja2; no checkpoint needed
just xcheck-pre # compare the pre-tokenizer against the regex GPT-2 and Llama 3 publish
just xcheck-quant # compare the dequantizers against ggml's block layouts; 14 formats
just xcheck-rope # compare RoPE against numpy's rotation from the RoFormer definition
just xcheck-fixture # parse gguftest's fixture with the Python gguf package and compare
just stress    # race binary under mixed load: plain, SSE, and clients that hang up; needs MODEL=
just fixture   # write a synthetic checkpoint big enough for just stress to interrupt
just skips     # count the tests that skip for want of a checkpoint, which CI reports too
just isolate   # run the suite twice with shuffling, to catch order dependence
just fmt       # gofmt -w, to fix whatever just check reported
just vet       # go vet on one package: just PKG=./internal/gguf vet
just clean     # remove bin/, .cache and the profiles; leaves the module cache alone
just release   # bump the version, write the changelog, tag; a person decides this, not CI
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

## Cross-checking the layers below the server

Four more checks are the same idea applied to the parts that feed the model, and none of them
needs a checkpoint:

| Recipe                | This side                     | The other side                            |
| --------------------- | ----------------------------- | ----------------------------------------- |
| `just xcheck-pre`     | `internal/tokenize` pre-split | GPT-2's and Llama 3's published regexes   |
| `just xcheck-quant`   | `internal/quant` dequantizers | ggml's `quantize.c` block layouts         |
| `just xcheck-rope`    | `internal/llama` RoPE         | numpy, from the RoFormer definition       |
| `just xcheck-fixture` | gguftest's synthetic model    | the Python `gguf` package reading it back |

The failure these exist for is one that produces plausible output rather than an error: a
nibble interleaving read the wrong way yields a scrambled matrix, and a wrong RoPE pairing
yields floats of the right magnitude. Both pass every test written from the same reading, which
is why each script writes its expectations from an outside source instead.

They all run in CI. Each writes its Go side to a temporary directory through a `go test -run`
inside the recipe, so nothing is left in the tree.

## Cross-checking the chat template

`just jinja-diff` renders a corpus of templates through `internal/chat` and through real Jinja2,
and diffs the text. It needs uv but no checkpoint: the corpus is in the script, and the Go side
is compiled at run time.

The same reasoning as above applies one layer up. The package's own tests assert against strings
typed by hand, which can share a blind spot with the code they check. Jinja2 is the reference
that does not.

A template using a construct outside the documented subset is expected to be **refused** rather
than rendered, and those cases are counted separately from disagreements, so "we declined to
evaluate this" does not read as a pass. The script's `SUBSET` table is the thing that decides
what the check can catch: a construct implemented but never given a case is one where a
regression goes unnoticed. `2605ec3` is the measured example -- `namespace()` shipped while
still listed as unsupported, so the tool exited 1 from that commit until a later one fixed the
table.

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
