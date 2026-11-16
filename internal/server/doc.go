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
// One limitation is deliberate and documented rather than hidden: a GGUF file
// stores its chat template as Jinja2, and this project will not take a Jinja
// dependency to evaluate it. The chat endpoint therefore renders a neutral
// "Role: content" transcript, and a caller who needs a model's exact template sends
// that prompt to the completions endpoint, which passes text through verbatim.
package server
