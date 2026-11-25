# Supported models and quantization

Orbix targets the GGUF ecosystem. The Status column names the release each family is planned to
ship in; no release has been tagged yet, so `v0.1` is the first one and `done` means implemented
and cross-checked against a real checkpoint rather than released.

| Family                  | Architectures        | Status        |
| ----------------------- | -------------------- | ------------- |
| **Llama / Llama 2 / 3** | `llama`              | v0.1          |
| **Qwen 2 / 2.5**        | `qwen2`              | v0.2          |
| **Qwen3**               | `qwen3`              | done          |
| **Mistral**             | `mistral`            | v0.3          |
| **Mixtral**             | `llama` with experts | v0.3          |
| **Gemma**               | `gemma`              | done          |
| **Gemma 2**             | `gemma2`             | done          |
| **Gemma 3**             | `gemma3`             | v0.4          |

| Quantization                   | Status |
| ------------------------------ | ------ |
| `F32`, `F16`                   | v0.1   |
| `Q8_0`                         | v0.1   |
| `Q4_0`, `Q4_1`                 | v0.2   |
| `Q5_0`, `Q5_1`                 | v0.3   |
| `BF16`                         | done   |
| `IQ4_NL`                       | done   |
| `K-quants` (`Q2_K`-`Q6_K`)     | done   |

GGUF parsing and tokenization are implemented in pure Go from the start; the quantization table
expands incrementally. ggml names more formats than this package decodes: fifteen of the weight
formats above have a block size in the size table and no decoder here -- `Q8_1` and `Q8_K`, which
are conversion-tool intermediates rather than formats a released checkpoint uses, and thirteen
of the newer low-bit and microscaling families (`IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS`, `IQ3_S`, `IQ4_XS`, `TQ1_0`, `TQ2_0`, `Q1_0`, `MXFP4`, `NVFP4`). A file carrying any of
them is refused with `cannot decode <type>` at load, the same way an unrecognised type is, and
`TestModelsDocListsEveryDecodableFormat` keeps this sentence and the tables honest in both
directions.

Six architectures are checked against real checkpoints: `llama` against llama.cpp's
log-probabilities, `qwen2` against a converted official checkpoint, `gemma` and `gemma2` against
llama.cpp's log-probabilities on converted Gemma 2B files, `qwen3` the same way on Qwen3-0.6B, and
`gemma3` on Gemma 3 270M. The MoE path has its own real file.

`gemma3` keeps Gemma 2's block -- sandwich norms, GeGLU, scaled embeddings -- and differs in three
ways that matter:

- Each head's query and key are normalized before the rotation. Gemma 2 does not, and its files carry
  no `attn_q_norm`/`attn_k_norm` tensors at all; adding the flag to `gemma2` was tried this round
  and reverted, because the loader then demanded tensors the files do not have.
- The attention softcap is gone. Gemma 2 bounds its scores with `attn_logit_softcapping`; Gemma 3's
  converter asserts the key is absent. The output softcap stays.
- The sliding window is per layer, and the file says which. Gemma 3 270M declares a window of 512
  and no pattern key, which llama.cpp reads as "every layer is local"; reading it as Gemma 2's
  "every second layer" leaves nine of eighteen layers attending over the whole context. The absence
  is only read that way for `gemma3`: Gemma 2's files predate the key and are described by the
  alternating rule, so applying one architecture's convention to the other bounded the wrong
  layers.

Measured on gemma-3-270m-it-Q8_0 against llama.cpp 0.3.37: the prompt encodes to the same six ids
and the greedy token agrees, with deviations of 0.0001, 0.66, 0.27, 0.08, 0.16, 0.42 and 0.66 nats
over the shared top-8 -- the low-probability tail, as with every checkpoint here.

**The window's boundary was off by one, and is fixed.** The paragraph above is measured on short
prompts, where the window never binds, so it could not see this. At a prompt length of 513 -- one
past gemma3's window of 512 -- this engine used to diverge from llama.cpp, and the divergence was not
a disagreement about the rule but about its boundary:

| prompt     | llama.cpp       | orbix, before   | orbix, after    |
| ---------- | --------------- | --------------- | --------------- |
| 512 tokens | `quick` -0.0000 | `quick` -0.0000 | `quick` -0.0000 |
| 513 tokens | `brown` -0.0000 | `fox`           | `brown` 42.015  |

The text is one sentence repeated, so the next token at each position is known: at 513 it is
`brown`. The window is a *distance*, not a count -- llama.cpp masks a key when
`pos_query - pos_key >= n_swa`, so the key at exactly `p - 512` is still visible, and a query at
position 512 reaches 513 positions including its own. `windowFor` returned the declared number as a
count, which dropped the oldest reachable position. One position of context, invisible while the
prompt fits in a window, and a different argmax once it does not.

What identified it was a numpy reference written directly against the mask rule, which could be
switched between `>` and `>=` on a prompt that distinguishes them. Five earlier attempts failed on
worse evidence, and one of them -- comparing every layer against the reference -- was itself wrong
because the reference's own block wiring had not been checked against a single layer's intermediates.
The block maths now has that check: `block()` reproduces layer 0 of gemma-3-270m-it-Q8_0 from the
embedding row to `max|diff| = 2.65e-04`, and the whole forward pass answers the same token and logit
(41.434) as this engine at the six-token calibration prompt.

Not implemented for `gemma3`: the separate rotation frequency for local layers. A file that declares
`gemma3.rope.freq_base_swa` is refused rather than rotated at the wrong rate. The 270M checkpoint
does not declare one, so it is unaffected.
`mistral` is not, and the reason is worth stating because it is not a gap this project can close by
working harder. The `mistral` architecture string is effectively retired upstream: the current
llama.cpp converter writes `general.architecture = llama` for every Mistral checkpoint it converts
(its Mistral class sets that explicitly "for compatibility"), and the arch enum in gguf-py no longer
contains `mistral` at all -- only `mistral3`, `mistral4` and `mistral-large4`. Four published
conversions were checked by reading their headers (TheBloke v0.1 instruct and base, second-state,
Mistral-Nemo) and every one declares `llama` with no sliding-window key. So there is no `mistral`
checkpoint in circulation to verify against, and this package's `mistral` entry describes a format
that files no longer use.

The sliding-window mechanism itself is verified on real weights anyway, through Gemma 2, whose config
does declare one (`TestWindowChangesRealLogits`): a window of 32 over 162 positions moves the logits
by 5.47 nats and changes the argmax, a window wider than the prompt leaves them bit-identical, and
bounding every layer differs from bounding every second one.

A real Mistral checkpoint was read as a check of the GGUF version 2 support instead, since that is
what it turned out to exercise.

`gemma` differs from `llama` in three ways that no packet listing shows, and each one was found
by comparing against llama.cpp rather than by reading the reference code:

- The rotary embeddings pair dimension `i` with `i + d/2`. Llama's files are converted with
  their query and key rows permuted so that the adjacent-pair rotation reproduces the original
  model, and Gemma's are not, so the pairing that looks right for Llama is wrong here. Measured
  on the same prefix, the adjacent pairing put the top-8 log probabilities 1.96 nats away and
  this one 0.19.
- Token embeddings are multiplied by `sqrt(hidden)` on the way in. The factor is applied at the
  lookup rather than folded into the matrix, because that matrix is usually the tied output
  projection and the projection has to see the unscaled weights.
- The feed-forward gate is GELU, not SiLU.

The reference implementation's norm layer multiplies by `(1 + weight)`, which invites a loader to
add the 1 at load time. This format does not need it: using the stored values directly reproduces
llama.cpp's log probabilities, while adding 1 puts them 26 nats away. The conversion was written,
measured, and removed.

`qwen3` is the smallest difference in the table and the easiest to get wrong: it keeps Qwen2's
rotation, drops the attention bias Qwen2 carries, and RMS-normalises every head's query and key
over the head width before rotating them. That step is invisible in the shapes -- the two norm
vectors are exactly the size of the two bias vectors they replace -- and the keys are normalised
where they are written, in four separate paths (single-token, batched prefill, batched decode, and
the cached query a reused prompt recomputes). Measured against llama.cpp on Qwen3-0.6B: the same
prompt ids, the same greedy token and the same top seven in the same order, with a worst deviation
of 0.29 nats; skipping the normalisation puts the same token 21 nats away.

`gemma2` adds four more, and one of them changes a shape rather than a number:

- Attention scores and output logits are each bounded by a softcap (`tanh(x/c)*c`), with the two
  values read from `attn_logit_softcapping` and `final_logit_softcapping`.
- Every block normalises its attention output and its feed-forward output before adding them to
  the residual. Without this the model is not recognisable at all: measured against llama.cpp,
  dropping both norms puts the top-8 log probabilities 130 nats away.
- Only every second block is bounded by the sliding window; the blocks between read the whole
  context. Applying the window to every layer is what the metadata key alone would do.
- The attention projections are not square. Gemma 2 2B has 2304 hidden and 8 heads but 256-wide
  queries and keys, so the per-head width comes from `attention.key_length` rather than from
  `hidden/heads`, which would give 288.

Measured on gemma-2-2b-it-Q3_K_L against llama.cpp: the same top-8 tokens in the same order, with
a worst deviation of 0.37 nats. That is larger than Gemma 1's 0.19 on a four-bit file, which is
what a three-bit quantisation costs when two implementations dequantize independently.

The softcaps cannot be checked against a real checkpoint this way, because at any ordinary
position the logits sit well inside their cap: turning both off moves the deviation by 0.004.
They are covered by constructed cases instead, at magnitudes where the cap binds -- a pair of
attention scores 200 and 100 with a cap of 50, and an output projection scaled until its logits
leave the +/-30 band -- and both answers are computable by hand. Writing those found a real bug:
the cap had been added to the single-token and batched-decode projections but not to the prefill
one, so the first token of every response came from a different distribution than the rest.

Quantization is checked at the block level rather than by comparing generations across formats:
exact-value tests on constructed blocks for the 32-weight formats, a round-trip error bound for
the K-quants, and a real tensor from a real checkpoint for both families. Comparing one prompt's
continuation across formats would test the model more than the decoder.

The constructed-block tests assert value by value, which is the only way to catch a permutation
-- swapping two nibbles reorders the weights and leaves every statistic untouched. The real-file
tests cover the other class: a decoder that reads a scale, a minimum or a quant field from the
wrong place still produces a well-ordered matrix, but one with a mean far from zero or a spread
off by orders of magnitude. Measured by breaking `decodeQ4_1`: reading the min slot as the scale
moved the mean to -0.512 and the spread to 0.518, both outside the bounds; the nibble swap was
caught only by the constructed-block test.

The 32-weight formats each have their own gate, because they live in different files and a
machine holding one should be able to check it. Any checkpoint carrying the type works as long as
its tensors are that type; the download below is 118-136 MB per format:

```bash
repo=RichardErkhov/ivnle_-_tinystories-lay8-hs512-hd8-33M-gguf
for q in Q4_1 Q5_0 Q5_1 IQ4_NL; do
  curl -sSL -o "/tmp/ts-$q.gguf" \
    "https://huggingface.co/$repo/resolve/main/tinystories-lay8-hs512-hd8-33M.$q.gguf"
done
MODEL_Q4_1=/tmp/ts-Q4_1.gguf MODEL_Q5_0=/tmp/ts-Q5_0.gguf \
MODEL_Q5_1=/tmp/ts-Q5_1.gguf MODEL_IQ4_NL=/tmp/ts-IQ4_NL.gguf \
CGO_ENABLED=0 go test ./internal/quant/ -run TestRealWeightMatrixStats -v
```

Measured on those four: 65,667,072 values each, standard deviation 0.0427 and a mean within
1e-5 of zero in all four -- the agreement a correct decoder should show across formats of the
same weights. `orbix run` on each produces the same text.

A file that carries none of the requested type fails rather than skips: the gate was set, so
something was meant to be checked, and a skip there would report success for a file that could
not answer the question.

TinyLlama 1.1B Q2_K exercises more of the engine: 22 layers, grouped-query attention (32 heads
over 4 KV heads), and a chat template the stories model lacks. It is also the file `MODEL_KQUANT`
wants, because it carries a Q6_K tensor:

```bash
curl -sSL -o /tmp/tinyllama-q2k.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf
```

Its Q6_K tensor decodes to a mean of -0.0004 and a standard deviation of 0.0247 over 65.5M
values -- the shape a weight matrix has. The count is what would go wrong first if a sub-block
layout were misread. Measured with that file: `mean -0.000390, std 0.024726` over `65536000`
values.

`scripts/hf_to_gguf.py` converts a Qwen2.5 checkpoint to GGUF for the weight-level cross-check,
deliberately independent of the Go code: a converter sharing the loader's assumptions could not
catch a wrong one. With
`QWEN2_MODEL` set it runs those tests:

```bash
# The converter reads the single-file form: config.json, model.safetensors, tokenizer.json.
mkdir -p /tmp/Qwen2.5-0.5B-Instruct && cd /tmp/Qwen2.5-0.5B-Instruct
for f in config.json model.safetensors tokenizer.json tokenizer_config.json generation_config.json; do
  curl -sSLO "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/$f"
done
cd - && uv run --with numpy python3 scripts/hf_to_gguf.py \
  /tmp/Qwen2.5-0.5B-Instruct /tmp/qwen2-f32.gguf
QWEN2_MODEL=/tmp/qwen2-f32.gguf CGO_ENABLED=0 go test ./internal/llama/ -run Qwen2
```

That checkpoint is about 1 GB and converts to a 2 GB F32 GGUF, so it is the largest of the four
gates. It is also the one with a visible end-to-end check: `orbix run` on the result answers
"The capital of France is" with "Paris." at 24 layers, 14 heads over 2 KV heads, and a 32768
position window. The file declares `chat_template`, so `serve` reports `chat: model template`
and the Qwen2.5 Jinja2 template renders the prompt.

Measured on a converted Qwen2.5-0.5B-Instruct, where the source directory was already on the
machine rather than downloaded (`uv run --offline --with numpy`, 3m22s wall for 290 F32 tensors):

```text
loaded 290 tensors / wrote 290 mapped tensors / vocab 151665 / 1979340608 bytes
orbix run --stats:  params 494032768, load 2.873s, gen 32 tokens in 1.726s, cache 38/32768
orbix serve:        chat: model template
/v1/chat/completions: content "Paris", finish_reason "stop", prompt_tokens 44
MODEL=… just xcheck: 290 tensors compared; all agree on type, shape and SHA-256
```

Two of those numbers are worth reading carefully, because both invite a wrong conclusion:

- **`vocab 151665` is not the vocabulary width.** `config.json` declares `vocab_size: 151936`, and
  `tokenizer.json` defines 151643 base pieces plus `added_tokens` up to id 151664. The converter
  fills the 271 undefined ids in between with `<unused{i}>` placeholders, so the file lists 151665
  pieces while the output projection stays 151936 wide. `model inspect` and `run --stats` both
  report 151936, which is the logits width.
- **`params 494032768` matches the safetensors header**, summed per tensor. The converter writes
  F32 throughout, so `model inspect` reports `1.84 GiB` for the file and the measured peak RSS was
  2,787,124 kB (2.66 GiB). That is `docs/configuration.md`'s rule -- resident memory follows the
  parameter count, not the file size -- holding on a real checkpoint, and the 1.4x expansion is
  small only because this file was already F32.

The `gen 32 tokens in 1.726s` figure is one observation on a machine whose load average moved
between 10 and 47 during the session; it is not a throughput claim.

`scripts/jinja_diff.py` renders chat templates through both this package and real Jinja2 and
reports differences. A disagreement is either a bug here or a construct the package documents
as unsupported; the tool separates the two and exits non-zero on any that render differently
rather than being refused.

```bash
# The built-in corpus: one template per family, plus one construct per case.
uv run --with jinja2 python3 scripts/jinja_diff.py

# A checkpoint's own template, read out of its metadata.
uv run --with jinja2 --with gguf python3 scripts/jinja_diff.py \
  --template-file /tmp/llama32.gguf
```

The two modes answer different questions and the first is weaker than it looks. The corpus is
hand-written: its `llama3` entry is one line, while the checkpoint's real template is 3827
characters that branch on tools, a date and a system message. So the corpus checks whether the
package implements a construct, and `--template-file` checks whether it can evaluate what a
model actually ships. Measured across Llama 3.2 1B, Qwen2.5 0.5B, a converted Qwen2.5 and
TinyLlama, 18 conversation shapes each: all identical to Jinja2.

The tool's own discriminating power was checked by breaking `trim` in `internal/chat`: the
corpus reports the precise failing construct, and `--template-file` reports the Llama 3.2
cases where the real template interpolates a `|trim`. A check that cannot fail proves nothing,
and this one was only ever run green before that experiment.

The MoE path gets the same treatment on a small public checkpoint; set `MIXTRAL_MODEL`. Its
author built it to validate MoE inference engines, and at 7 MB it is the cheapest of the four
gates to fetch. It declares 4 experts at 2 used per token, on the `llama` architecture:

```bash
curl -sSL -o /tmp/tinymoe2m.F32.gguf \
  https://huggingface.co/shibatch/tinymoe2m/resolve/main/tinymoe2m.F32.gguf

MIXTRAL_MODEL=/tmp/tinymoe2m.F32.gguf CGO_ENABLED=0 go test ./internal/llama/ -run TestMoEReachesTheExpertsOnARealModel
```

Any mixture-of-experts GGUF this loader accepts works; the test reads `expert_count` from the
file rather than assuming a shape, so it is not tied to this one.

The stories model is a 19 MB download and the only one of the four needing no conversion:

```bash
curl -sSL -o /tmp/stories15M-q4_0.gguf \
  https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf
```

Every gate taken together leaves no test skipped, so a machine holding the checkpoints can
check every claim the project makes. `go test ./... -v | grep -c '^--- SKIP'` is the count, and
it is worth re-measuring rather than trusting these figures -- adding a gated test raises it.
Measured 2026-11:

| Gates set                                                                                                         | Skipped |
| ----------------------------------------------------------------------------------------------------------------- | ------- |
| none                                                                                                              | 31      |
| the six that were on hand (`MODEL`, `QWEN2_MODEL`, `QWEN3_MODEL`, `GEMMA2_MODEL`, `GEMMA3_MODEL`, `MODEL_KQUANT`) | 9       |
| all fourteen                                                                                                      | 0       |

Both end figures re-measured 2026-11 with `go test -v ./...`; the middle row was measured the same
way on 2026-11-25, with six checkpoints downloaded from ModelScope and set. The nine it leaves are
worth naming because they show what the remaining gates buy: three are the `ORBIX_*` dump switches
(`TestDumpRope`, `TestDumpDequant`, `TestDumpPreSplit`), which read no checkpoint at all, and six
need a file those six do not cover -- Gemma 1, a MoE checkpoint, a Llama v2 file, a FIM/code model,
one of the four 32-weight quantizations, and the reference-corpus generator. A machine that has six
of the fourteen is therefore closer to covered than the ratio suggests.

The table names three rows rather than a full ladder because a row per gate would mostly restate
the same fact. What a reader needs is where the count starts, that it reaches zero, and -- from the
middle row -- that most of the distance is covered by the small checkpoints rather than only by the
large ones.

There are fourteen gates: ten named directly at the call site, and four that
`internal/quant/formats_real_test.go` looks up from a table. The four table rows are easy to miss
because nothing at a call site reads them.

| Variable        | Checkpoint                   | Covers                        |
| --------------- | ---------------------------- | ----------------------------- |
| `MODEL`         | `stories15M-q4_0.gguf`       | generation, batching, tracing |
| `QWEN2_MODEL`   | a converted Qwen2 file       | `qwen2`                       |
| `QWEN3_MODEL`   | `Qwen3-0.6B-Q4_K_M.gguf`     | `qwen3`                       |
| `MISTRAL_MODEL` | a version 2 GGUF             | GGUF v2 parsing               |
| `GEMMA_MODEL`   | `gemma-2b.gguf`              | `gemma`                       |
| `GEMMA2_MODEL`  | `gemma2.gguf` (Q3_K_L)       | `gemma2`, sliding window      |
| `GEMMA3_MODEL`  | `gemma-3-270m-it-Q8_0.gguf`  | `gemma3`, prompt token ids    |
| `MIXTRAL_MODEL` | `tinymoe2m.F32.gguf`         | the MoE path                  |
| `MODEL_KQUANT`  | `tinyllama-q2k.gguf`         | K-quant decode                |
| `MODEL_Q4_1`    | a Q4_1 file                  | `Q4_1` decode                 |
| `MODEL_Q5_0`    | a Q5_0 file                  | `Q5_0` decode                 |
| `MODEL_Q5_1`    | a Q5_1 file                  | `Q5_1` decode                 |
| `MODEL_IQ4_NL`  | an IQ4_NL file               | `IQ4_NL` decode               |
| `CODER_MODEL`   | `Qwen2.5-Coder-0.5B` GGUF    | FIM prompt token ids          |

The four quantization ones are easy to miss when auditing this list: they are not `os.Getenv("...")`
at the call site but rows in a table that `TestRealWeightMatrixStats` walks, so a search for
`Getenv("` finds ten gates and the four rows are invisible. Measured both ways -- 31 skipped with no
variables, 0 with all fourteen.

That search is worth trusting only after measuring, because the two answers look equally plausible.
Setting those ten leaves exactly one skip, `TestRealWeightMatrixStats`, which is the one test the
four table rows feed. So ten gates do not reach zero and fourteen do, and the count in the table
above is the one that was measured rather than derived from a grep.

That paragraph has now been wrong once in each direction, which is worth saying because it is the
whole point of keeping the count. It first said ten, then sixteen, then thirteen; each figure was
believed until someone ran the command. The last correction added a gate twelve hours after the table
was written: `CODER_MODEL`, for the FIM prompt check, which the count did not know about because it
was added in the same round as the paragraph that enumerated the others.

`MISTRAL_MODEL` covers the format revision rather than an architecture: the first real Mistral
checkpoint reached this project was GGUF **version 2**, and the file declares `llama` with no sliding
window. It is the test that would have caught the refusal of every version 2 file in circulation.

```bash
MODEL=/tmp/stories15M-q4_0.gguf \
QWEN2_MODEL=/tmp/qwen2-f32.gguf \
QWEN3_MODEL=/tmp/qwen3.gguf \
MISTRAL_MODEL=/tmp/mistral.gguf \
GEMMA_MODEL=/tmp/gemma-2b.gguf \
GEMMA2_MODEL=/tmp/gemma2.gguf \
GEMMA3_MODEL=/tmp/gemma3-q8.gguf \
MIXTRAL_MODEL=/tmp/tinymoe2m.F32.gguf \
CODER_MODEL=/tmp/qwen-coder.gguf \
MODEL_KQUANT=/tmp/tinyllama-q2k.gguf \
MODEL_Q4_1=/tmp/ts-Q4_1.gguf \
MODEL_Q5_0=/tmp/ts-Q5_0.gguf \
MODEL_Q5_1=/tmp/ts-Q5_1.gguf \
MODEL_IQ4_NL=/tmp/ts-IQ4_NL.gguf \
CGO_ENABLED=0 go test ./...
```

This block used to set eight variables while the table above listed fourteen, so a reader who ran it
got a skip and no explanation. The count in the table is what `go test -v ./... | grep -c '^--- SKIP'`
reports with every one of these set, and it has been re-measured after each gate was added.

`MODEL_KQUANT` is the one path that must not be reused casually. Its test skips when the file
holds no K-quant tensor, so pointing it at an F32 or Q4_0 file reports a skip rather than a
failure, and a reader counting skips sees the gate "pass" having decoded nothing. Measured:
`MODEL_KQUANT=/tmp/qwen2-f32.gguf` -- the Qwen2 file, which is all F32 -- skips with "no K-quant
tensor found in the model". TinyLlama Q2_K above is the file to use; it carries a Q6_K tensor.

`just test` does not set these variables, so the gated tests skip under it exactly as they do
without them. Writing the block out here is what a reader holding the checkpoints needs; putting
the paths in the recipe would mean every contributor's `just test` ran a different suite.

An untrained file produces a flat next-token distribution, so its output is gibberish whatever
the implementation does. The MoE test therefore asserts that zeroing the experts changes the
logits -- a statement about the code, not about the file.

# Checking the engine against a second implementation

`internal/llama/reference_test.go` compares this engine to a numpy forward pass on fixtures the
numpy side did not produce. The fixtures in `internal/gguftest` are written and read by this
repository, so a wrong assumption about a dimension order, a block layout, or where a tensor's
bytes begin is invisible to every test built on them -- the same code makes the same mistake in
both directions. A second implementation written from the architecture description shares no code
with this one, so a disagreement points at one of the two rather than at a shared premise.

The expected values live in `internal/llama/testdata/reference/logits.json` and are checked against
six fixtures, each pinning one thing a reference can get independently wrong:

| Fixture          | What it pins                                             |
| ---------------- | -------------------------------------------------------- |
| `llama`          | RMSNorm, adjacent-pair rope, GQA, SiLU gate              |
| `gemma-sandwich` | scaled embeddings, sandwich norms, GeGLU                 |
| `qwen3`          | per-head QK norm combined with NeoX rope                 |
| `qwen2-bias`     | the q/k/v attention biases, which are tensors not shapes |
| `windowed`       | a window of 4 against a twelve-token prompt              |
| `window-edge`    | a window of 4 against five tokens -- the boundary itself |

The two windowed ones are not redundant. The sliding window's off-by-one survived every fixture
test in the repository for several rounds *because* every prompt was shorter than every window, so
the fixture that exceeds its window is what makes the boundary observable at all; `window-edge`
then puts the boundary on the last position, where the two readings differ by the largest fraction
of the reachable set. Measured under the same injected fault, `windowed` disagrees by 0.154 logits
and `window-edge` by 0.227.

The six disagreements the test currently shows are 1.9e-06 to 5.7e-06 logits, against a tolerance
of 1e-3. That gap is float64 accumulation on the numpy side against float32 here, not slack in the
check: the faults injected while writing it measured 0.154 and 0.227 (window reach), 0.219 (rope
pairing), and three that only the newer fixtures can see -- `qwen2`'s attention bias removed,
`qwen3`'s QK norm removed, and `gemma2`'s GELU gate replaced with SiLU. Each fails its own fixture
and no other.

## Regenerating the expected values

The numpy side is not in this repository, because a Go project does not need a Python dependency
to build or test. To reproduce it, write the fixtures and this engine's logits with:

```bash
ORBIX_REFERENCE_DIR=/tmp/golden \
  go test -run TestGenerateReferenceInputs -v ./internal/llama/
```

That writes one `.gguf` and one `.bin` per spec -- the fixture, and this engine's logits in a flat
format a numpy script can read without a JSON parser -- plus `engine.json` with the same numbers.
Have a numpy implementation of the same architecture reproduce the logits from the `.gguf` files,
then write its results into `testdata/reference/logits.json` as
`{"<spec>": {"ids": [...], "logits": [...]}}`. The test compares this engine against that file, so
what it checks is the numpy side; `engine.json` is there to make a disagreement easy to locate. What matters is that the numpy side is written from
the architecture rather than translated from this engine; a translation reproduces the same
mistakes and the check becomes decorative.

Two things the numpy side has to get right, both found by it disagreeing:

- The rope pairing is per architecture. `llama` and `mistral` rotate adjacent pairs (2k, 2k+1);
  the Qwen and Gemma families rotate k with k + d/2. Using one formula for both made the `llama`
  fixture disagree by 0.219 logits while the argmax stayed correct -- a model that answers the same
  token from the wrong numbers.
- Whether to apply `attn_q_norm` is an architecture property, not a tensor's presence. The
  `gemma2` fixture carries the tensor and the architecture does not use it; a reference keyed on
  whether the tensor exists applied it and disagreed by 0.342.

## What a failure tells you

The test reports the first block whose residual disagrees, not only the final logits:

```text
the divergence begins at layer 1: worst 1.313309 at dimension 26 (engine -133.841248, reference -132.527939)
layers 0..0 agreed; everything after this is downstream of it
```

That line is the difference between a test that reports a symptom and one that reports a location.
The sliding window's off-by-one took six rounds to find largely because nothing said which layer
disagreed, so each hypothesis had to be tested by rebuilding the comparison by hand. The per-layer
residuals are stored in `testdata/reference/logits.json` alongside the logits for the same reason the
window fixtures exist: the cheap-looking addition is what makes the expensive case tractable.

Only the last prompt token's residuals are stored. Earlier positions have their own, and a fault in
one of them shows up as a fault in the last one anyway, so including them would multiply the file by
the prompt length for no additional signal.

When every layer agrees the message says so, which is itself informative: a disagreement that
appears only in the final logits is in the output norm or the vocabulary projection, and those are
two places rather than the whole model.

## The same comparison on real weights

`TestMatchesNumPyReferenceOnRealWeights` runs the comparison on a checkpoint instead of a fixture.
It is gated on `GEMMA3_MODEL` and `GEMMA3_REFERENCE`:

```bash
GEMMA3_MODEL=/tmp/gemma3-q8.gguf GEMMA3_REFERENCE=gemma3-270m.json \
  go test -run TestMatchesNumPyReferenceOnRealWeights -v ./internal/llama/
```

`GEMMA3_REFERENCE` is a file name under `testdata/reference`, not a path: the expected values are
part of the repository and a path would let a run compare against something outside it.

Fixtures carry small pseudo-random weights, so two implementations that agree on them agree on the
shape of the arithmetic and little else. A checkpoint brings the numerical range they cannot:
residual streams in the thousands rather than below ten, norms whose weights were trained rather
than set to one, and a 262144-wide vocabulary whose projection dominates the error budget. That is
also why this test compares relative error against a tolerance of 1e-3 while the fixture test uses
absolute logits -- 1e-3 absolute against a residual of 5109 would be asking float32 for twelve
significant figures.

Six reference files are checked, one per property none of the others can see:

| File               | Checkpoint   | What only this one can see                               |
| ------------------ | ------------ | -------------------------------------------------------- |
| `gemma3-270m.json` | gemma-3-270m | the numerical range of trained weights                   |
| `gemma3-513.json`  | gemma-3-270m | the sliding window binding past position 512             |
| `mistral-v2.json`  | Mistral v0.1 | GGUF **version 2** and K-quant tensors                   |
| `gemma-1.json`     | Gemma 2B     | a block with no QK norm and no sandwich norms            |
| `qwen2.json`       | Qwen2.5-0.5B | q/k/v attention biases with NeoX rope                    |
| `qwen3.json`       | Qwen3-0.6B   | per-head QK norm on a prompt whose tokenizer adds no BOS |

The two Qwen files are the only ones with NeoX rope outside the Gemma family, and between them they
separate the two features that travel together elsewhere: `qwen2` has the biases without the QK norm,
`qwen3` the QK norm without the biases. A reference that pairs them -- applying one because the other
is present, or keying either on the architecture's family rather than its spec -- disagrees with one
of these two while agreeing with every Gemma file.

`qwen3.json` also covers a tokenizer difference rather than an arithmetic one: its prompt is five
tokens with no BOS, where the same text under Qwen2 produces six with one. Reading the ids from the
file would have hidden that; both files store their ids, and the difference between them is visible
in the two files side by side.

`mistral-v2.json` is the only one that loads a version 2 file. Gemma 3 changed nothing about the
containers this project reads, so a regression in the version check would not show up in any Gemma
file -- and would show up here as a load failure rather than as wrong numbers. Its K-quant tensors
also come from `gguf.quants` on the reference side rather than a decoder written for this project:
the block layouts are intricate enough that reimplementing them would be an invitation to copy the
thing being checked.

`gemma-1.json` is the negative case for two features. Every other checkpoint normalises each head's
query and key, or applies sandwich norms, or both; Gemma 1 does neither, so it is what fails if
either gets applied unconditionally.

Measured: on both, the three sampled layer residuals agree, the top eight logits agree to within
1e-3 relative, and the argmax is the same token. On the 513-token file that argmax is `brown`, which
is what llama.cpp answers too -- an independent third opinion, since the numpy reference shares no
code with either.

Injected faults this catches include a 1% error in the embedding scale (6.7e-03 relative, reported
against the specific logits that exceed tolerance) and, on the 513-token file only, the sliding
window's off-by-one: that fault moves `brown` from 42.01 to 39.14, a relative error of 6.8e-02, sixty
times the tolerance. The six-token file is blind to it because six tokens is far shorter than a
window of 512. This is why the `window-edge` fixture and the 513-token checkpoint file both exist:
the fixture catches the boundary in a few milliseconds, the checkpoint confirms the same boundary at
real magnitudes.

A repeated prompt is stored as `prompt_repeat` rather than as 513 ids, and the test re-tokenizes it
with the checkpoint's own tokenizer. Listing the ids would make the file mostly prompt, and copying
them would skip the tokenizer -- which is one of the things being checked.

Only the top eight logits and three layer residuals are stored, 37 KiB rather than 5 MiB. The
informative part of a 262144-wide distribution is its head, and the first, middle and last blocks are
enough to say where a divergence starts.
