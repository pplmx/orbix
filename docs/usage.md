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

`./models/` is not in the repository, so `--model` has to point at a checkpoint you already have.
Anything the architecture table in [Models](models.md) lists will serve; the download snippets on
that page name a file per family, and stories15M q4_0 (19 MB) is enough to watch a request finish.

The server implements `/v1/chat/completions`, `/v1/completions` (streaming and not),
`/v1/models` plus `/v1/models/{id}` for `models.retrieve`, `/healthz` and `/metrics`. An
unimplemented path under `/v1` answers 404 in the API's JSON envelope rather than Go's
plain-text one, because the SDKs decode the body before raising.

`/healthz` answers `{"status": "ok", "model": "<name>"}` and `/metrics` serves the Prometheus
exposition. Both are answered from the HTTP layer without touching the engine, which is what
makes them useful as probes and is worth being precise about:

- **Ready means loaded.** `serve` creates the engine and only then calls `net.Listen`, so a
  connection to `/healthz` cannot succeed before the weights are resident. Measured on the
  553 MiB Llama 3.2 checkpoint: 6.3 s from process start to the first 200.
- **Healthy does not mean unloaded.** Neither endpoint reports engine state. Under a saturated
  KV pool -- `orbix_kv_pool_free` at 0 with four requests holding every block -- `/healthz`
  still answered 200 in 0.5 ms and `/metrics` in 0.7 ms. A liveness probe wants that; a load
  balancer looking for "skip the busy instance" has to read `/metrics` itself, because this
  server will accept the request and queue it rather than report itself unavailable.
- **A failed `/healthz` means the process is gone**, not that a model is missing: the model is
  loaded before the socket exists, so that failure happens at startup and exits rather than
  showing up as an unhealthy instance later.

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

A body sent compressed without being decompressed is the one answer whose status is not 400:
`Content-Encoding: gzip` on a request this server will not expand is refused with
`415` and `code: unsupported_encoding`, naming the encoding it read. The distinction matters to
a client because the body was never parsed -- the fix is to send it uncompressed, and a 400 with
`invalid_json` would send a reader looking through bytes that were fine.

Not every error has one. A refused request answers 503 with `type: service_unavailable_error`
and no `code`, because there is nothing in the request to correct -- retry, or back off. Read a
missing `code` as "this answer is about capacity, not about your fields", and branch on `type`
for the retryable case. Treating the absence as a malformed response would make a correctly
shaped 503 look like a proxy error.

Unsupported fields split by what a client does with the answer, and the split is not the one this
section originally drew. `logit_bias`, `presence_penalty` and `frequency_penalty` were listed as
ignorable sampling knobs and are now honoured: a caller who bans a token or asks for less repetition
is asking for different text, not expressing a tolerance, and all three measured byte-identical
output with and without them. Fields describing the *response* are refused with a 400 naming every
one, since a client that sent `n: 3` indexes `choices[2]`.

`response_format` is refused for a third reason: `json_object` promises valid JSON, which needs
constrained decoding this engine does not implement, and honouring the request without it is not
possible. Measured on Gemma 3 270M, a reply asked for as JSON came back as a markdown fence, so a
client calling `json.loads` on it fails somewhere far from the request that caused it. `{"type":
"text"}` and an absent field are served, because they ask for what the server does anyway.

Chat's `echo` is still refused this way: OpenAI's chat schema does not define it, so there it
describes a field the caller sent that the endpoint cannot use.

Function calling used to be the other, refused in every spelling a client might use. It is
implemented now, and the model's own template is what renders it:

- `tools` reaches the prompt through the checkpoint's own tool branch, so Qwen2.5 and Llama 3.1
  each see the signatures in the format they were trained on. A model whose template this project
  cannot evaluate refuses tools with a 400 rather than accepting and dropping them.
- A reply carrying the call marker comes back as a `tool_calls` array -- `id`, `type: "function"`,
  and a `function` with `name` and `arguments` as a JSON **string**, which is the schema's shape
  and what every SDK unmarshals -- with `finish_reason: "tool_calls"`. Arguments that are not a
  JSON object are left in the text rather than passed on: a parser that guessed would hand a
  client a value that fails in its own code, far from the response.
- A conversation's `tool_calls` on an assistant turn and its `role: "tool"` results are rendered
  back, so a multi-turn exchange works. `tool_call_id` is checked against the ids the conversation
  actually declared.
- Streaming holds back exactly the bytes that could still become a call, so a streamed reply never
  shows the marker or the JSON as content; the parsed call arrives on the terminal chunk.

Three things are still refused, each with a reason rather than a blanket "not implemented":
`tool_choice: "required"` and a named `tool_choice` (no template in the supported subset has a
field that forces a call, so the constraint would reach no prompt), `function.strict` (it promises
the arguments match the schema exactly, which needs constrained decoding this engine does not
have), and the deprecated `functions`/`function_call` pair (a different schema; translating it
would drop fields while looking like it worked). Empty forms are served, because they ask for what
the server would do anyway -- `tools: []` and `tool_choice: "none"` answer normally.

Everything else that used to be refused has been implemented instead -- `n`, `logprobs`,
`top_logprobs`, `echo` on completions, `max_completion_tokens`, `best_of` and `suffix` -- each
leaving the list when it did. A field belonging to the other endpoint is not refused there: it is
an unknown member and is ignored, so `suffix` sent to chat is silently dropped rather than named.

The alternatives inside `top_logprobs` are ordered by decreasing probability, and that order is
preserved on the wire. The schema calls the field an object, which invites a map, and any Go map is
encoded with its keys sorted -- so an implementation built that way returns the alternatives
alphabetically while looking correct in an editor. It matters because the order carries the meaning:
the first entry is the model's top prediction, and a client that takes it as such is reading what
llama.cpp's servers return. Measured on Gemma 3 before this was fixed, the most likely token of
"The capital of France is" came third in the encoded object.

`logprobs` is honoured on both, in each endpoint's own shape: completions answers with parallel
arrays (`tokens`, `token_logprobs`, `text_offset`, `top_logprobs`), chat with a list of objects
carrying `token`, `bytes`, `logprob` and `top_logprobs`. The count of alternatives differs by
endpoint because OpenAI documents different maxima -- `logprobs` at most 5 on completions,
`top_logprobs` at most 20 on chat -- and a count past either is refused with `invalid_value`
naming the bound rather than clamped. Streaming carries a logprobs object on each delta chunk,
and the streamed and non-streaming lists are equal for the same request, offsets included.

One consequence of those shapes is worth knowing before reading `top_logprobs` on completions. Its
alternatives are a JSON object keyed by token text, so two alternatives that render to the same
string become one entry -- and the count that comes back can be shorter than the count that was
asked for. It is not a rare vocabulary defect: a token whose bytes are not valid UTF-8 on its own
renders as U+FFFD, and a byte-level vocabulary has one per continuation byte. Measured,
Qwen3-0.6B has 1138 token ids that render as a single U+FFFD and stories15M has 129. Measured on a
small fixture with 128 such ids: a request asking for 5 alternatives came back with 4 keys, and two
of the values were out of descending order because the survivor carried the later position's number.
The chosen token's own probability is reported separately in `token_logprobs` and is unaffected.

Chat's shape does not share this: its alternatives are a list of objects each carrying `bytes`, so
nothing collapses and the requested count always comes back. A caller that needs the full list on a
byte-level vocabulary, or needs to tell two same-rendering tokens apart, should read chat's
`top_logprobs` instead.

The probabilities come from a log-softmax over the whole vocabulary at the request's
temperature, not from the truncated set a `top_k` or `top_p` setting leaves. That distinction
matters: describing the truncated set would report a logprob of 0 whenever `top_k` is 1, which
says "the model was certain" about a position where the request merely narrowed the choice. A
greedy request (`temperature: 0`) reports the distribution at temperature 1, since there is no
divisor to apply.

`best_of` on completions generates that many candidates and returns the best. "Best" is the
highest mean log probability per token -- the mean rather than the sum, because every term is
negative and a sum always prefers the shortest completion.

"Per token" means per token the response reports, which after a `stop` string is not the same as per
token the candidate generated. A candidate that runs past its stop string has those tokens cut from
the response, and scoring them would let it win on text the caller cannot see. That is not
hypothetical: with `stop: "."` and temperature 2, the returned choices came back in an order that
rose and fell, because the candidate that had generated thirty tokens past its stop was ranked on all
of them. `shownTokens` in `internal/server/bestof.go` is the code that draws this line, and it is the
place to look when a ranking looks wrong. With `n`, `best_of` is the candidate
count and `n` is how many to return, in ranked order with `index` counting from 0; a `best_of`
that is not greater than `n` is refused rather than ignored, since it asks for no selection at
all. `stream: true` is refused for the same reason OpenAI documents: the candidates can only be
compared once they have finished, so there is nothing to stream until the choice is made. The
`usage` figures count every candidate generated, not only the returned ones, because the work
was done either way and OpenAI's own description warns that `best_of` consumes quota.

`suffix` on completions turns the request into an insertion: `prompt` is the text before the
insertion point and `suffix` the text after, and the reply is what belongs between them. That
needs a checkpoint trained for it, so the server looks for the marker names a family uses --
`<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>` for Qwen and CodeGemma, the same names
without the pipes for StarCoder, `<PRE>` / `<SUF>` / `<MID>` for CodeLlama, and the SentencePiece
spelling of the first set with the word-boundary space (`\u2581<fim_prefix>` and its two
siblings), which is how a vocabulary that encodes a leading space as a separate marker writes
them -- and refuses with
`unsupported_parameter` if the served model declares none of them, naming the sets it looked for
so an operator can see what the checkpoint would have needed. Generating from the prefix alone
would be worse than the refusal: it is a plausible continuation that silently ignores half of what
the caller sent.

The assembled prompt matches llama.cpp token for token on a real code model. Measured on
Qwen2.5-Coder-0.5B whose prefix ends in an indented `return` and whose suffix opens two new function
definitions, both produce the same eighteen ids, the markers coming through as single special tokens
rather than as text. That distinction is worth stating because the same check written without llama.cpp's special-
token flag agrees with the wrong answer: it splits `<|fim_prefix|>` into nine ordinary pieces and
still produces a plausible-looking list, so a comparison that skipped the flag would pass while
testing the tokenizer rather than the markers.

What the model does with a correct prompt is its own business. The same file answered that prefix and
suffix with `return a * b` -- multiplication inside `add` -- and then carried on writing further
functions, and llama.cpp produces the identical text from the identical prompt. A 0.5B AWQ
quantization is not a code model to judge an implementation by; the markers are checked here, the
completions are not.

Declaring the markers is necessary but not sufficient. A vocabulary comes from the tokenizer a
checkpoint was converted with, so an instruct model can carry a code model's markers without
having been trained to use them -- measured on Qwen2.5-0.5B-Instruct, which declares all three,
answers an assembled fill-in-the-middle prompt by repeating the suffix rather than filling it.
Nothing in a GGUF file distinguishes the two cases, so the server trusts what the file says.

The insertion ends at the marker the model emits to close it, and which one that is belongs to
the checkpoint: the three framing markers join the stop set, and so does any dedicated closing
marker the vocabulary declares -- `<|fim_pad|>` and `<|endoftext|>` for Qwen, `<fim_pad>` for
StarCoder, `<EOT>` for CodeLlama. Leaving those out is not a detail: measured on
Qwen2.5-Coder-0.5B with only the framing markers, a prefix ending in an indented `return` with
a one-newline suffix produced `x * x` and then carried on writing a whole `main()` function,
because nothing told it the insertion was over. With the closing markers in the list the same
request stops at the end of the expression. A checkpoint that emits neither will run to
`max_tokens`, so `stop` is still worth setting on an unfamiliar model. `suffix` with an empty string is not an insertion and is served normally, and `suffix` with
`echo` is refused: with an insertion the prompt is only the text before the insertion point, so
echo would return part of what was sent.

A token whose bytes are not valid UTF-8 on their own -- one half of a character, in a byte-level
vocabulary -- is described by its bytes, which is why chat's schema has that field: `bytes` is
authoritative and `token` is its rendering. Concatenating `bytes` over the entries reconstructs
the message content exactly.

`max_completion_tokens` is honoured on both: it is OpenAI's newer name for the same limit, and
`max_tokens` is deprecated in favour of it, so a migrated client sends only the newer one. Sending
both is refused with `invalid_value` naming both fields and both numbers, because they are
alternatives rather than a pair and the two values can disagree; picking one silently would
answer with a budget the caller may not have meant.

A budget of zero is honoured, not read as "unset": the response is empty with `finish_reason`
`"length"`, which is what a caller who computed a budget of zero asked for. Omitting the field is
what takes the server's default. The two used to be the same thing -- the field decoded to a plain
integer, so an explicit zero silently became the default length, and a request for nothing came back
with a full paragraph. The types distinguish them now.

OpenAI defines the field on chat only -- the official SDK's completions parameters do not include
it, so no SDK client sends it there. This server honours it on completions as a deliberate
superset: treating it as an unknown member and ignoring it, which is the rule for unknown fields,
would generate the default length for a caller who reached for a limit, and nothing in the
response would say so.

`echo: true` on completions returns the prompt ahead of the generated text rather than refusing
it. Two things about it are worth knowing. The stop list applies to the generated part only, so a
stop string that appears in the prompt does not truncate the prompt; the echoed text is exactly
what the caller sent. And the echoed prompt is not counted in `completion_tokens` -- it is already
in `prompt_tokens`, and counting it twice would bill the same text once per field. That last point
is this server's choice rather than a documented rule: the schema does not say, and the reading
that keeps `prompt_tokens + completion_tokens` equal to the tokens actually processed is the one
taken. Streaming sends the prompt as a delta before any generated token, once per choice.

`n` is not on either list: the server generates the choices it asks for. Every choice is a
separate sequence in the same batch, so `n: 3` costs one round per token rather than three
requests, and the body carries `choices[0..n-1]` with `index` set to the position. `usage`
counts `completion_tokens` across all of them and `prompt_tokens` once, which is what the
figures describe: the prompt was prefilled once per sequence but is the same text. With a
`seed`, choice *i* is seeded with *seed + i*, so the same request returns the same set in the
same order.

**The choices differ from each other only when sampling is on.** Greedy decoding is
deterministic by definition, so at the default `temperature: 0` every choice takes the same
argmax and `n` copies of one completion come back -- measured, `n: 4` at the default returned the
same six tokens four times, identically across three requests. That is not a bug in the seed
handling: the per-choice seeds are distinct, but a greedy sampler never reads one. Set
`temperature` above zero for `n > 1` to mean anything, and note the same applies to `best_of`,
whose candidates are only distinguishable once sampling is on -- measured, `best_of: 4, n: 2` at
`temperature: 0` returned two identical choices, and at `temperature: 0.9` two different ones.

One limit remains on `n`, and it is reported rather than silent: it must be at most 8, since
each choice holds its own KV blocks and they are all in flight at once. The refusal names the
bound, so a caller learns what would have been accepted.

`stream: true` works with `n > 1` too. The chunks are interleaved as the choices advance, and
every chunk's `index` says which choice its delta belongs to, so a client accumulates per index
and gets `n` texts. Each choice gets its own finish chunk carrying its `finish_reason`, because
`[DONE]` arrives once for the whole stream and is not a per-choice signal. With
`stream_options.include_usage`, the usage chunk totals every choice the same way the
non-streaming body does. Measured through the OpenAI Python SDK with `n=2, max_tokens=6`: two
interleaved texts, two `finish_reason: "length"` chunks, one `[DONE]`, and
`completion_tokens: 12`, `prompt_tokens: 4`.

The refusal turns on the **value**, not on the field being present. A field sent with the value
the response already carries asks for nothing this server would not have done anyway, so it is
served: `suffix: ""` answers 200, and `logprobs: false` asks for no log probabilities, which is
what the response does. That matters because the official SDKs serialise a keyword argument the
caller passed even when it equals the API default, so `completions.create(suffix="")` puts
`"suffix": ""` on the wire. Refusing on presence told such a caller about a request it serves
correctly; measured through the OpenAI Python SDK, four such calls went from 400 to 200.

Most of what used to illustrate this rule is now implemented rather than refused -- `n`,
`logprobs`, `top_logprobs`, `echo`, `max_completion_tokens`, `best_of` and `suffix` all work,
and each left this list when it did. Chat's `echo` is the one field still refused.

`suffix` is not refused as a field; it is refused for the checkpoints that cannot use it. A
model declaring none of the known fill-in-the-middle markers answers 400 naming the sets that
were looked for, because generating from the prefix alone would drop half of what the caller
sent. On a code-tuned checkpoint the same request is served -- see
[the insertion rules](#serve-a-gguf-model) above.

A value the API does not define is reported as malformed rather than unimplemented: `n: -1` is
400 `invalid_value` naming the number, not `unsupported_parameter` describing `n: 3`. The two ask
different things of the caller. A negative count has to change whatever the server supports,
while an unimplemented field has to be dropped, and saying the wrong one sends them to edit the
wrong part of the request.

`stream_options.include_usage` is honoured: it is the only way a streaming caller learns what it
spent. The extra chunk carries the counts before `[DONE]` and matches the non-streaming body.

[`internal/server/doc.go`](../internal/server/doc.go) is the reference for the full contract -- the
complete code list, the field-by-field rules, and the stop-string semantics. Two things about the
chat endpoint are worth knowing before pointing a client at it:

- **The chat template is the model's own, when it can be evaluated.** GGUF stores a
  checkpoint's format as a Jinja2 string under `tokenizer.chat_template`, and
  `internal/chat` evaluates a subset of it: `if`/`elif`/`else`, `for` with the `loop`
  variables, `set` scoped the way Jinja2 scopes it (a `for` body is a frame, an `if` body is
  not), `+` (which adds two integers and joins anything else), `-`, `%`, comparisons, the `is`
  tests `defined`, `none`, `mapping`, `iterable` and `string`, and the `trim`, `tojson` and
  `length` filters. That covers the
  published templates for Qwen2.5 and Llama 3.1 exactly, and both are checked against their
  real source: `scripts/jinja_diff.py --template-file <checkpoint.gguf>` reads
  `tokenizer.chat_template` out of the file and renders it through this package and through
  real Jinja2. Measured on Llama 3.2 1B (a 3827-character template that branches on tools, a
  date and a system message), Qwen2.5 0.5B (2507), a converted Qwen2.5 (2509) and TinyLlama
  (410): 18 conversation shapes each, every one rendering identically.
  A template using syntax outside the subset is reported at parse time; one calling a
  function this package does not implement fails at render time instead, because the parser
  accepts any call. Either way the server falls back to a neutral `Role: content`
  transcript. The fallback matters: a half-rendered template would produce a prompt that
  looks plausible and is wrong, while a neutral transcript is merely less good than the
  model's own format.

  Both kinds are named in the startup banner, which probes the template with one throwaway
  conversation before choosing its line: `serve` prints `chat: neutral transcript (reason)`
  and `chat` prints the same. Measured with a template of `{{ messages[99].content }}`, the
  banner reads `chat: neutral transcript (chat: render: index 99 outside a sequence of 1)`
  rather than claiming the model's template is in use.

  A template's own variables are a third case, and they are honoured: `chat_template_kwargs` is
  the vLLM-spelled member that sets them, so a field like Qwen3's `enable_thinking` reaches the
  template. Values are the ones JSON carries -- booleans, numbers, strings, null, and lists or
  objects of those -- and they render as Jinja2 would render what `json.loads` gives it, so an
  integer prints `3` and a fraction `2.5` rather than every number printing as a float.

  Three cases are refused with a 400 naming `chat_template_kwargs` and the variable, because each
  would otherwise produce a prompt that ignored what the caller sent:

    - a model whose template this project cannot evaluate, where no variable can reach anything;
    - a name the template never reads;
    - a name the renderer binds itself: `messages`, `bos_token`, `eos_token`,
      `add_generation_prompt`, `tools`.

  This is the opposite of how an unknown member is treated, and the distinction is the one the
  rest of this page draws: an unknown member asks for nothing the server claims to provide, while
  a variable this model cannot act on is a request whose effect the caller will look for in the
  reply and not find. The refusal names the variable, so a client learns which one to drop.

  A template that survives the probe and then fails on a real conversation still exists, and
  only `serve` can see it, so it reports one `WARN` record on the first such conversation,
  carrying the same reason. It is logged once rather than per request, since such a template
  fails every time, and it is suppressed when the banner already named the fallback -- the
  caller sets that flag, because only the caller knows what it printed.
  If you would rather control the prompt, send it to `/v1/completions`, which passes
  text through verbatim.

- **`stop` is matched on decoded text, but only a single-token string stops generation
  early.** A four-token string like `"中国的首都"` truncates correctly and still runs to its
  budget unread; `"\n"` encodes to one token and halts the sequence itself.

  The truncation is exact across token boundaries, in streaming and not, and for text that is not
  ASCII. Measured on Gemma 3 270M with a prompt whose reply continued `首都，也是世界最古老的城市之一。`:
  `stop: ["世界最古老"]` returned `首都，也是` and `stop: ["城市"]` returned `首都，也是世界最古老的`, both with
  `finish_reason: "stop"` and both valid UTF-8 -- the cut lands on a character boundary even when the
  stop string straddles two tokens, because the cut is applied to decoded text rather than to a
  byte offset into a token buffer. The prompt is never searched: `stop: ["colors"]` against a prompt
  containing "colors" answered normally. Streaming agrees with the non-streaming body for the same
  requests, including a stop string that spans two chunks.

- **`stop` takes any number of strings, not OpenAI's four.** The documented limit there is
  `maxItems: 4`; this server accepts a hundred and honours all of them -- measured, a request
  whose only match was in the fifth slot stopped at it. A client that enforces four is not
  wrong, it is just stricter than it needs to be, and nothing here rejects a longer list.
  Matching costs nothing measurable: 1000 stop strings measured the same 0.26 s as none, because
  the cut runs on the decoded text after generation rather than per token.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain continuous batching."}],
    "stream": true
  }'
```

`model` is left out on purpose, and it has to be: a server serves exactly one checkpoint and refuses
any other name with `model_not_found` naming the one it does serve. An absent field means "the only
one there is", which is what a single-model server should do with a request that does not name it.
This example used to send `"model": "local"`, which was a name no server had -- copying it produced
a 404 that said so, from a command written to be copied.

## Behind a reverse proxy

The server sets `X-Accel-Buffering: no` on streaming responses, which is the one thing it can
do from its side of the connection. nginx buffers proxied responses by default
(`proxy_buffering on`), so without that header an SSE stream still arrives complete but all at
once, which for a token stream is the same as not streaming. nginx reads the header and turns
buffering off for that response only; every other endpoint keeps the buffering that a proxy
default is actually for. Proxies that do not understand the header ignore it.

Three things remain the deployment's job, because no application header can express them:

- **Raise `proxy_read_timeout`.** nginx defaults to 60 seconds bounding the gap between reads,
  which a long model load or a stalled generation can exceed.
- **Keep reading.** A stream nobody reads is abandoned, not paused, once the backpressure
  reaches the engine: 64 events buffer per sequence, then two seconds of grace, then the slot
  goes to someone else. The bound is on the engine's queue rather than on the network, and
  kernel socket buffers absorb a surprising amount, so the guarantee is "drain as you go" --
  which every SSE client does by default -- not "you have two seconds".
- **Use HTTP/1.1 upstream.** nginx defaults to 1.1 from 1.29.7; before that it is 1.0, where
  ending a stream means closing the connection.

A shutdown is bounded too, and this one is visible to every streaming client. `SIGTERM` (or
`SIGINT`) stops accepting new requests, lets in-flight responses finish, and only then stops
the pump -- so an ordinary stream completes normally, `[DONE]` included. The drain waits at
most `shutdownGrace`, which is five seconds. A response still running when it expires is cut,
and what the client receives depends on the endpoint:

- **A stream** ends mid-token and **without a `[DONE]` sentinel**. That is the signal: every SSE
  client already treats a stream that stops before `[DONE]` as incomplete, and it is the one
  honest thing this server can say, because the OpenAI schema has no `finish_reason` for "the
  server went away". Measured on Llama 3.2 1B with `max_tokens: 4000` and a 1-second grace: 50
  delta chunks, a final chunk with `finish_reason: "length"`, and then the connection closes
  with no sentinel. The same request with no shutdown was still streaming after 120 seconds with
  994 chunks.

  The finish chunk still carries `"length"`, and that part is not a claim about the token budget:
  the cut comes from the server cancelling the request, the engine finishes the sequence with
  reason `cancel`, and `cancel` is reported as `"length"` because nothing else in the schema fits.
  Read the absence of `[DONE]`, not the finish reason. Earlier versions sent the sentinel too,
  which made a cut stream byte-for-byte identical to a complete one.

  A client that hung up instead of being cut sees the same reason but nobody is reading, so the
  distinction costs nothing there.

- **A non-streaming request** receives `503` with a body explaining that the request was
  abandoned before it finished. Earlier versions closed the connection with no response at
  all, which a client cannot tell apart from a crash: measured, four concurrent non-streaming
  requests against four slots at a 3-second grace all died as `curl` exit 52, "empty reply from
  server". They now all answer 503, and the same four at a 60-second grace still answer 200.

The server logs `shutdown: a response was still in flight when the 5s grace expired; it was cut`
to stdout either way, so the cut is visible on the server side even when the client learns
nothing.

Measured two ways. On Llama 3.2 1B at `--max-tokens 500`, `SIGTERM` two seconds in waited the
full 5606 ms, delivered 75 chunks, and closed the connection with no `[DONE]`. On stories15M
with a stream that finishes in about 20 ms, the same signal exited in 21 ms with all 125 chunks
and their `[DONE]` intact -- which is the drain doing its job rather than the cut.

A client cannot distinguish that cut from a dropped connection, so a deployer who drains a pod
should treat a stream that ends without `[DONE]` as incomplete rather than as finished.

The effective value is printed at startup, so it is visible before a signal arrives:

```text
shutdown:  5s grace for in-flight responses, then they are cut
```

Five seconds is not enough for a real generation, so `--shutdown-grace` sets it. Measured on
this engine, a 64-token reply takes 5.04 s on Llama 3.2 1B, 5.17 s on TinyLlama 1.1B and 3.41 s
on Qwen2.5-0.5B; 128 tokens on Llama 3.2 takes 10.4 s. An ordinary reply from a modest model
therefore sits at or past the default, and a rolling restart under load will cut it.

The flag is worth setting in one direction or the other:

```bash
# A service whose clients retry: restart fast and let them.
orbix serve --model ./model.gguf --shutdown-grace 0

# A batch service that would rather finish the reply than restart promptly.
orbix serve --model ./model.gguf --shutdown-grace 60s
```

Measured on the same request -- Llama 3.2 1B, 40 tokens, `SIGTERM` one second in: at
`--shutdown-grace 1s` the server exited in 1591 ms with 19 chunks and no `[DONE]`; at `30s` it
drained in 3205 ms with all 42 chunks and their `[DONE]`. At `0` it exited in 676 ms with 9
chunks and logged `the 0s grace expired`. A negative value is refused, because
`context.WithTimeout` with a negative duration is already expired -- every shutdown would cut
every response while claiming to drain.

**A grace long enough for one request is not long enough for a full batch.** Concurrent
requests share one engine, so their remaining time is the batch's, not one reply's. Measured:
eight concurrent 128-token streams on Llama 3.2 1B were all cut after the full 30-second grace
(816 chunks, 0 of 8 with `[DONE]`). Those 816 chunks are 102 tokens per stream in 30 seconds, so
each stream advanced at 3.4 tok/s -- a quarter of the 12 tok/s one stream gets alone -- and 128
tokens would need about 38 seconds at that rate -- and with a 60-second grace that batch did
finish, in 38571 ms, all eight with their `[DONE]`. The same batch at 32 tokens per stream
finished inside a 180-second grace in 9.0 seconds.

So the value to set is what `--max-active` in-flight replies need from the model, not what one
reply needs. Per-stream throughput is roughly the shared throughput divided by the active
stream count, so the drain a batch needs grows with both `--max-tokens` and `--max-active`.

A deployment that cannot leave that much time -- a Kubernetes `terminationGracePeriodSeconds`
defaults to 30 -- should lower `--max-tokens`, reduce `--max-active` during deploys, or accept
that the last batch is cut and have clients retry. The last is the case `--shutdown-grace 0`
makes explicit rather than surprising: a process that does not wait cannot truncate anything
half-way through waiting for it.

**`--shutdown-grace` must fit inside the orchestrator's own grace period, not the other way
round.** Kubernetes sends `SIGTERM`, waits `terminationGracePeriodSeconds` (30 by default), then
`SIGKILL`. A `--shutdown-grace` longer than that is never reached: measured with `SIGTERM` and a
`SIGKILL` three seconds later, the process died at exit 137 with 46 chunks delivered and no
`[DONE]`, and the `shutdown: ... grace expired` line was never printed because the process was
gone before its own timer fired. That is worse than a clean cut, and it is silent.

That it is the *ordering* of the two numbers, and not the length of the generation, is what a
controlled pair shows. Same synthetic model, same 18000-token stream, same SIGTERM and SIGKILL
three seconds apart; only `--shutdown-grace` differs. At `30s` the process is killed before its
own timer fires: no `[DONE]`, no line in the log, nothing to tell an operator the reply was cut.
At `1s` the server cuts the stream itself and says so -- `shutdown:  1s grace for in-flight
responses, then they are cut`. The cut is the same in both; only one of them admits to it.

So keep `--shutdown-grace` below `terminationGracePeriodSeconds`, leaving enough of the
difference for the process to act on its own deadline, or raise the pod's grace to match what
the batch needs. The two values are one decision, made in two files.

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

## In a container

The repository ships a `Dockerfile` that builds the binary and puts it in a `scratch` image.
That is possible because the build is statically linked: measured, `ldd` says "not a dynamic
executable" and the ELF has no dynamic section, so the runtime needs no libc and no shell.

```bash
docker build -t orbix --build-arg VERSION=v0.1.0 --build-arg COMMIT="$(git rev-parse --short HEAD)" .
docker run --rm -p 8080:8080 -v /path/to/models:/models:ro \
  orbix --model /models/your-model.gguf
```

The image does not contain a model. A checkpoint is hundreds of megabytes to several gigabytes
and belongs in a volume or a derived image; the default `CMD` names `/models/model.gguf` as a
placeholder so that a run without a `--model` fails with a path rather than starting a server
with nothing to serve.

Two details in that file are easy to get wrong, and both were found by building from the exact
context `.dockerignore` produces rather than from the working tree:

- **`-buildvcs=false` is required.** `.dockerignore` excludes `.git`, and without the flag the
  go command stops with `error obtaining VCS status: exit status 128`. Version metadata comes
  from `--build-arg` instead, because a build context has no reliable `.git` and stamping a wrong
  commit hash is worse than stamping none.
- **The CA bundle is copied in.** Trace export is OTLP over HTTP, which means TLS to any real
  collector, and `scratch` has no certificate store -- so without it every span fails with
  `x509: certificate signed by unknown authority` and only `orbix_trace_spans_dropped_total`
  shows it.

### Probing it

`scratch` has no curl, no wget and no shell, so both probe mechanisms need an http client that is
already in the image. `orbix healthcheck` is that client, and it is what the image's
`HEALTHCHECK` line runs:

```bash
orbix healthcheck --addr http://127.0.0.1:8080          # prints ok and the model name
orbix healthcheck --addr http://127.0.0.1:8080 --quiet  # exit status only
```

It checks more than the status code: the body's `status` must be `"ok"`, because a 200 from a
proxy or a stale process on the same port is "something is listening" rather than "this model is
ready". The same command works as a Kubernetes exec probe, which avoids needing an `httpGet`
probe to reach a server that may be bound to loopback:

```yaml
livenessProbe:
  exec:
    command: ["/orbix", "healthcheck", "--addr", "http://127.0.0.1:8080", "--quiet"]
  periodSeconds: 30
  timeoutSeconds: 5
  failureThreshold: 3
readinessProbe:
  exec:
    command: ["/orbix", "healthcheck", "--addr", "http://127.0.0.1:8080", "--quiet"]
  periodSeconds: 10
  timeoutSeconds: 5
  # The engine is built before net.Listen, so a 200 means the weights are resident. A checkpoint
  # that takes longer than this to load will fail readiness until it finishes, which is what the
  # probe is for -- but a large model on slow storage can exceed a minute.
  initialDelaySeconds: 60
```

`terminationGracePeriodSeconds` has to be larger than `--shutdown-grace`, not the other way
round; [Behind a reverse proxy](#behind-a-reverse-proxy) measures what happens when it is not.

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
