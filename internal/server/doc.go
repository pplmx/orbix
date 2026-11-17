// Package server is the HTTP layer: it turns JSON requests into engine
// submissions and engine events back into JSON or server-sent events.
//
// It is the last layer in the reading order (GGUF, kernels, KV cache, scheduler,
// server) and it is deliberately the thinnest. Everything that makes inference
// work -- tokenization, batching, cache management -- is already decided below it;
// what is left is the part that only exists because the caller is on the network:
// decoding a request body, choosing a status code, flushing a stream, and noticing
// that a client hung up.
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
// The one thing this layer adds beyond transport is /metrics, because exporting the
// engine's counters is a job that belongs to whoever owns the process lifetime. The
// numbers themselves are collected in internal/engine: a server that inferred queue
// depth or pool occupancy from the outside would be reporting a guess. What lives
// here is only the encoding, and it is written by hand -- four lines of Prometheus
// text format do not justify a client library in a repository whose point is to make
// each layer legible.
package server
