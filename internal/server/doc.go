// Package server is the HTTP layer: it turns JSON requests into engine
// submissions and engine events back into JSON or server-sent events.
//
// It is the last layer in the reading order (GGUF, kernels, KV cache, scheduler,
// server) and it makes no inference decisions of its own. Everything that makes
// inference work -- tokenization, batching, cache management -- is already decided below
// it; what is left is the part that only exists because the caller is on the network:
// decoding a request body, refusing a field the engine cannot honour, choosing a status
// code, flushing a stream, and noticing that a client hung up.
//
// "Makes no inference decisions" is a statement about the work, not about the size: this
// package has the fourth-largest number of functions in the repository, behind tokenize,
// llama and chat, because the OpenAI schema is wide and refusing unsupported fields one by
// one is most of it. By lines it is also fourth, a little behind chat -- there are many
// small functions here and few long ones, which is what a schema-shaped layer looks like
// next to a decoder. The thinness that matters is that none of those lines can change what
// the model computes.
//
// The API shape is OpenAI's, which is the de-facto interface for local inference.
// Implementing it means the server can be driven by the OpenAI client libraries and
// by editor plugins rather than by a bespoke protocol, and it means the layer can
// be checked against a client that was not written for it.
//
// The chat endpoint renders a checkpoint's own format: a GGUF file stores its chat
// template as Jinja2 under tokenizer.chat_template, and internal/chat evaluates the
// subset of it that published templates use. A template outside that subset is reported
// at parse time and the endpoint falls back to a neutral "Role: content" transcript,
// which the server prints at startup so the choice is visible. A caller who wants to
// control the prompt exactly sends it to the completions endpoint, which passes text
// through verbatim.
//
// A request-validation error carries a "code": invalid_json, unsupported_encoding and
// body_too_large for a body the decoder rejected, invalid_type for a field whose JSON type
// is wrong, and invalid_value, missing_required_parameter, unsupported_parameter,
// context_length_exceeded and model_not_found for a request it understood and refused.
// OpenAI's guidance is to branch on "code" rather than on message text, and paramError
// exists to keep the code paired with the field that produced it. "param" is present when
// one field is at fault and absent when the failure is about the body as a whole.
//
// invalid_type is separate from invalid_json because the two ask different things of the
// caller: a body that did not parse has to be re-sent, while a field of the wrong type has to
// be edited, and the answer names which field -- {"max_tokens": "8"} is 400 invalid_type with
// param max_tokens. Folding the second into the first was the earlier behaviour, and an SDK
// branching on code read "your JSON is broken" for a body that parsed.
//
// A chat message's content is OpenAI's union: a string, or an array of {"type","text"}
// parts. Both are accepted and their text is what reaches the prompt. A part of any other
// type -- an image_url, say -- is refused with unsupported_parameter and
// param messages.content rather than dropped, because answering the text of a message that
// also carried an image would reply to a question the caller did not ask.
//
// Capacity and internal failures have no code. A full queue answers 503 with
// service_unavailable_error and a message, and a broken stream likewise: there is no field
// for a caller to fix, and the retryable signal the SDKs read is "type", not "code". A client
// must therefore treat a missing code as "nothing to correct in the request" rather than as a
// malformed response -- which is what the earlier wording here claimed, saying every error
// body carried one.
//
// Unsupported request fields are handled by what a client does with the answer. A field
// that only tunes sampling -- logit_bias, presence_penalty, response_format -- is ignored,
// because the SDKs send such fields whether or not the engine can use them; the text is not
// what was asked for, but it is the shape the client reads. A field that describes the
// response is refused with a 400 naming it: a client that sent n: 3 indexes choices[2], so
// answering with one choice is not a smaller response but a different one, and the failure
// would surface as an IndexError far from the request that caused it.
//
// The refused set differs per endpoint, because the two schemas do: chat refuses n, logprobs,
// top_logprobs and echo, while completions refuses n, logprobs, echo, suffix and best_of.
// max_completion_tokens is refused on both, for a sharper reason than the rest -- it replaces
// max_tokens rather than accompanying it, so a migrated client sends only the newer name and
// would otherwise get the default length with nothing saying so.
//
// What decides a refusal is the value, not the field's presence. A field sent with the value
// the response already carries is a request this server serves: n: 1, best_of: 1, echo: false,
// logprobs: false, top_logprobs: 0 and suffix: "" all answer 200. The official SDKs serialise
// a keyword argument the caller passed even when it equals the API default, so
// completions.create(n=1) sends "n": 1 -- and refusing on presence answered it with
// "n>1 needs several sequences", which is a sentence about a different request.
//
// A value the API does not define is malformed rather than unimplemented: n: -1 answers 400
// invalid_value naming the number. The distinction is whose decision it is. A negative count
// has to change whatever this server implements, while a unimplemented field has to be dropped,
// and naming the wrong one sends the caller to edit the wrong part of the request.
//
// stream_options.include_usage is honoured rather than ignored, because it is the only way a
// streaming caller learns what it spent: without it a streamed response reports no usage at
// all. With it, one extra chunk arrives before [DONE] whose choices is an empty array and
// whose usage holds the whole request's counts, and every ordinary chunk in the same stream
// carries "usage": null. The counts equal what the non-streaming body reports.
//
// A stop string is matched on decoded text, so a multi-token string truncates correctly; what
// it does not do is stop generation sooner. The sequence runs to its budget unread while the
// response is already cut. A stop that encodes to one token ("\n", "###") is handed to the
// engine as an id, so that one halts the sequence itself.
//
// The one thing this layer adds beyond transport is /metrics, because exporting the
// engine's counters is a job that belongs to whoever owns the process lifetime. The
// numbers themselves are collected in internal/engine: a server that inferred queue
// depth or pool occupancy from the outside would be reporting a guess. What lives
// here is only the encoding, and it is written by hand -- a three-line family in a
// format whose whole grammar fits on a page does not justify a client library in a
// repository whose point is to make each layer legible.
package server
