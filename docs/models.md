# Supported models and quantization

Orbix targets the GGUF ecosystem. Initial support:

| Family                  | Architectures        | Status        |
| ----------------------- | -------------------- | ------------- |
| **Llama / Llama 2 / 3** | `llama`              | v0.1          |
| **Qwen 2 / 2.5**        | `qwen2`              | v0.2          |
| **Mistral**             | `mistral`            | v0.3          |
| **Mixtral**             | `llama` with experts | v0.3          |
| **Gemma**               | `gemma`              | not supported |

| Quantization                   | Status |
| ------------------------------ | ------ |
| `F32`, `F16`                   | v0.1   |
| `Q8_0`                         | v0.1   |
| `Q4_0`, `Q4_1`                 | v0.2   |
| `Q5_0`, `Q5_1`                 | v0.3   |
| `BF16`                         | done   |
| `IQ4_NL`                       | done   |
| `K-quants` (`Q2_K`-`Q6_K`)     | done   |

GGUF parsing and tokenization are implemented in pure Go from the start; the quantization table expands incrementally. Two ggml types are defined and sized but not decoded -- `Q8_1` and `Q8_K` -- and both are conversion-tool intermediates rather than formats a released checkpoint uses. A file carrying either is refused with `cannot decode <type>` at load, the same way an unrecognised type is.

Two of the three architectures are checked against real checkpoints: `llama` against
llama.cpp's log-probabilities, `qwen2` against a converted official checkpoint. The MoE path has
its own real file. `mistral` does not -- its configuration and sliding window are covered against
constructed files.

Quantization is checked at the block level rather than by comparing generations across formats:
exact-value tests on constructed blocks for the 32-weight formats, a round-trip error bound for
the K-quants, and a real K-quant tensor when `MODEL_KQUANT` is set. Comparing one prompt's
continuation across formats would test the model more than the decoder.

TinyLlama 1.1B Q2_K exercises more of the engine: 22 layers, grouped-query attention (32 heads
over 4 KV heads), and a chat template the stories model lacks:

```bash
curl -sSL -o /tmp/tinyllama-q2k.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf
```

Its Q6_K tensor decodes to a mean of -0.0004 and a standard deviation of 0.0247 over 65.5M
values -- the shape a weight matrix has. The count is what would go wrong first if a sub-block
layout were misread.

`scripts/hf_to_gguf.py` converts a Qwen2.5 checkpoint to GGUF for the weight-level cross-check,
deliberately independent of the Go code: a converter sharing the loader's assumptions could not
catch a wrong one. With
`QWEN2_MODEL` set it runs those tests:

```bash
uv run --with numpy python3 scripts/hf_to_gguf.py ./Qwen2.5-0.5B-Instruct /tmp/qwen2-f32.gguf
QWEN2_MODEL=/tmp/qwen2-f32.gguf CGO_ENABLED=0 go test ./internal/llama/ -run Qwen2
```

`scripts/jinja_diff.py` renders checkpoint templates through both this package and real Jinja2
and reports differences. A disagreement is either a bug here or a construct the package
documents as unsupported; the tool separates the two and exits non-zero on any that render
differently rather than being refused.

```bash
uv run --with jinja2 python3 scripts/jinja_diff.py
```

The MoE path gets the same treatment on a small public checkpoint; set `MIXTRAL_MODEL`:

```bash
MIXTRAL_MODEL=/tmp/tinymixtral.gguf CGO_ENABLED=0 go test ./internal/llama/ -run TestMoEReachesTheExperts
```

The stories model is a 19 MB download and the only one of the four needing no conversion:

```bash
curl -sSL -o /tmp/stories15M-q4_0.gguf \
  https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf
```

The four gates take the suite from fifteen skipped tests to none, so a machine holding the
checkpoints can check every claim the project makes: `go test ./... -v | grep -c '^--- SKIP'`
reads 15 without a checkpoint and 0 with all four set. Re-measure rather than trusting the
number -- adding a gated test raises it.

```bash
MODEL=/tmp/stories15M-q4_0.gguf \
QWEN2_MODEL=/tmp/qwen2-f32.gguf \
MIXTRAL_MODEL=/tmp/tinymixtral.gguf \
MODEL_KQUANT=/tmp/tinymixtral.gguf \
CGO_ENABLED=0 go test ./...
```

`just test` does not set these variables, so the gated tests skip under it exactly as they do
without them. Writing the block out here is what a reader holding the checkpoints needs; putting
the paths in the recipe would mean every contributor's `just test` ran a different suite.

An untrained file produces a flat next-token distribution, so its output is gibberish whatever
the implementation does. The MoE test therefore asserts that zeroing the experts changes the
logits -- a statement about the code, not about the file.
