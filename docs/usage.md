# Usage

## Serve a GGUF model

```bash
CGO_ENABLED=0 go install github.com/pplmx/orbix/cmd/orbix@latest

orbix serve \
  --model ./models/qwen2.5-7b-instruct-q4_0.gguf \
  --addr :8080
```

The prefix is not decoration: `go install` uses whatever CGO_ENABLED the environment has, which
is 1 wherever a C compiler exists, and such a binary reports `cgo: true` in `orbix version`. The
code has no `import "C"`, so that build works, but it is not the artifact this project ships.

The server implements `/v1/chat/completions`, `/v1/completions` (streaming and not),
`/v1/models` plus `/v1/models/{id}` for `models.retrieve`, `/healthz` and `/metrics`. An
unimplemented path under `/v1` answers 404 in the API's JSON envelope rather than Go's
plain-text one, because the SDKs decode the body before raising.

A request-validation error carries a `code`, and `param` names the field at fault when there is
one. Branch on `code`, not on message text -- OpenAI's guidance, and the reason is that the codes
cover the bodies a client hits before any field parses (`invalid_json`, `body_too_large`) as well
as the ones it understands and refuses (`context_length_exceeded`, `unsupported_parameter`).

`invalid_type` is its own code, with the field in `param`, because the fix differs from a body
that did not parse: `{"max_tokens": "8"}` has to be edited rather than re-sent, and the answer
says which field and which type arrived. A chat message's `content` accepts both spellings
OpenAI defines -- a string, or an array of `{"type": "text", "text": ...}` parts -- and a part of
any other type, an image for instance, is refused with `unsupported_parameter` on
`messages.content` rather than silently dropped.

Not every error has one. A refused request answers 503 with `type: service_unavailable_error`
and no `code`, because there is nothing in the request to correct -- retry, or back off. Read a
missing `code` as "this answer is about capacity, not about your fields", and branch on `type`
for the retryable case. Treating the absence as a malformed response would make a correctly
shaped 503 look like a proxy error.

Unsupported fields split by what a client does with the answer. Sampling knobs -- `logit_bias`,
`presence_penalty`, `response_format` -- are ignored, because the SDKs send them regardless.
Fields describing the *response* are refused with a 400 naming every one, since a client that
sent `n: 3` indexes `choices[2]`.

The refused set is not the same on the two endpoints, because their schemas differ. Chat refuses
`n`, `logprobs`, `top_logprobs` and `echo`; completions refuses `n`, `logprobs`, `echo`, `suffix`
and `best_of`. A field belonging to the other endpoint is not refused there -- it is an unknown
member and is ignored, so `suffix` sent to chat is silently dropped rather than named.
`max_completion_tokens` is refused on both, for a sharper reason: it replaces `max_tokens` rather
than accompanying it, so a migrated client would silently get the default length.

`stream_options.include_usage` is honoured: it is the only way a streaming caller learns what it
spent. The extra chunk carries the counts before `[DONE]` and matches the non-streaming body.

[`internal/server/doc.go`](../internal/server/doc.go) is the reference for the full contract -- the
complete code list, the field-by-field rules, and the stop-string semantics. Two things about the
chat endpoint are worth knowing before pointing a client at it:

- **The chat template is the model's own, when it can be evaluated.** GGUF stores a
  checkpoint's format as a Jinja2 string under `tokenizer.chat_template`, and
  `internal/chat` evaluates a subset of it: `if`/`elif`/`else`, `for` with the `loop`
  variables, `set`, `+` concatenation, comparisons, the `is` tests `defined`, `none`,
  `mapping` and `iterable`, and the `trim`, `tojson` and `length` filters. That covers the
  published templates for Qwen2.5 and Llama 3.1 exactly, and both are checked against their
  real source: `scripts/jinja_diff.py --template-file <checkpoint.gguf>` reads
  `tokenizer.chat_template` out of the file and renders it through this package and through
  real Jinja2. Measured on Llama 3.2 1B (a 3827-character template that branches on tools, a
  date and a system message), Qwen2.5 0.5B (2507), a converted Qwen2.5 (2509) and TinyLlama
  (410): 18 conversation shapes each, every one rendering identically.
  A template using syntax outside the subset is reported at parse time; one calling a
  function this package does not implement fails at render time instead, because the parser
  accepts any call. Either way the server falls back to a neutral `Role: content`
  transcript, printing which of the two it is using at startup. The fallback matters: a
  half-rendered template would produce a prompt that looks plausible and is wrong, while a
  neutral transcript is merely less good than the model's own format.
  If you would rather control the prompt, send it to `/v1/completions`, which passes
  text through verbatim.
- **`stop` is matched on decoded text, but only a single-token string stops generation
  early.** A four-token string like `"中国的首都"` truncates correctly and still runs to its
  budget unread; `"\n"` encodes to one token and halts the sequence itself.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Explain continuous batching."}],
    "stream": true
  }'
```

## Behind a reverse proxy

The server sets `X-Accel-Buffering: no` on streaming responses, which is the one thing it can
do from its side of the connection. nginx buffers proxied responses by default
(`proxy_buffering on`), so without that header an SSE stream still arrives complete but all at
once, which for a token stream is the same as not streaming. nginx reads the header and turns
buffering off for that response only; every other endpoint keeps the buffering that a proxy
default is actually for. Proxies that do not understand the header ignore it.

Two things remain the deployment's job, because no application header can express them:

- **Raise `proxy_read_timeout`.** nginx defaults to 60 seconds bounding the gap between reads,
  which a long model load or a stalled generation can exceed.
- **Keep reading.** A stream nobody reads is abandoned, not paused, once the backpressure
  reaches the engine: 64 events buffer per sequence, then two seconds of grace, then the slot
  goes to someone else. The bound is on the engine's queue rather than on the network, and
  kernel socket buffers absorb a surprising amount, so the guarantee is "drain as you go" --
  which every SSE client does by default -- not "you have two seconds".
- **Use HTTP/1.1 upstream.** nginx defaults to 1.1 from 1.29.7; before that it is 1.0, where
  ending a stream means closing the connection.

`X-Accel-Buffering` is nginx-specific and consumed rather than forwarded (it is in the default
hidden-header set), so a two-tier nginx chain needs `proxy_pass_header X-Accel-Buffering` on the
inner tier. HAProxy and Envoy forward bytes as they arrive and need nothing; Caddy flushes
`text/event-stream` regardless. All of them still need timeouts sized for a response that can
outlive the default.

## No CORS and no authentication

The server sends no `Access-Control-Allow-*` headers, so a browser page on another origin
cannot read a response -- and because the request body is JSON decoded without a Content-Type
check, a page can still *send* one. `POST` with `Content-Type: text/plain` is a CORS "simple
request", which the browser delivers without a preflight, and the server parses the body
whatever the header says. The page cannot see the reply, but the request runs:

```bash
# Accepted from any origin, without a preflight, from a page the user merely visited.
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: text/plain' -H 'Origin: http://example.invalid' \
  -d '{"prompt": "...", "max_tokens": 256}'
```

This is a property of an unauthenticated API that also performs work, not a bug in the CORS
decision: CORS governs who may read a response, not who may cause one, and checking the
Content-Type would break the `curl -d` form above. What closes it is binding to loopback or
authenticating in front -- `--addr :8080` listens on every interface by default.

Request bodies are bounded to `ContextLen` x 48 bytes with a 1 MiB floor -- 6 MiB at 128k
context. That is not a tuning knob so much as the context constraint seen from the request side:
a larger body cannot describe a prompt the model could attend over. Anything over is answered
`413` without reading the rest; unbounded, a 64 MiB request spent 82 seconds being read and
tokenized before the server could say it was too long.

## Embedding it in a Go service

Not supported today, and the reason is structural rather than an omission. Every package that
would do the work lives under `internal/`, which Go forbids another module from importing:

```text
use of internal package github.com/pplmx/orbix/internal/engine not allowed
```

That is deliberate for now -- `internal/` is what makes the package boundaries a rule the compiler
enforces rather than a convention, and the layers have moved enough that a frozen public API would
have been a liability. The interfaces in [Core interfaces](architecture.md#core-interfaces) are
the shape such an API would take; the table there says which of them exist and which do not, and
nothing in this section should be read as a promise that they are reachable.

What works instead:

- **Run the server and talk HTTP.** `orbix serve` is the supported integration surface, and it is
  the OpenAI shape, so the client libraries already exist: see
  [Serve a GGUF model](#serve-a-gguf-model), and the `curl` calls there are the contract.
- **Fork or copy what you need.** The code is structured to be read, and `internal/engine` is the
  entry point a fork would call: `engine.New(path, engine.Options{...})` returns an engine,
  `Submit` queues, `Pump` advances a round, and `Stats` reports. That is what the CLI does.

## Cross-compile

```bash
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o orbix-arm64 ./cmd/orbix
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o orbix.exe ./cmd/orbix
GOOS=linux GOARCH=riscv64 CGO_ENABLED=0 go build -o orbix-riscv64 ./cmd/orbix
```

`CGO_ENABLED=0` is repeated on each line rather than exported once above them, because these are
lines a reader copies one at a time and a forgotten export is invisible until the binary reports
`cgo: true`.

No toolchain setup beyond Go itself. The version is not incidental: `go.mod` declares `go 1.27.0`,
and the code uses generic methods and `strings.CutLast`, both of which arrived in that release, so
an older toolchain fails to build rather than degrading quietly.
