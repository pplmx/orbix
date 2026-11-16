// Package engine ties the model together with request scheduling: it owns the
// loaded weights, hands out per-sequence KV caches, and turns a stream of
// prompts into a stream of tokens.
//
// A scheduling round has two phases. Every sequence that still holds prompt
// tokens prefills in one batched pass; every sequence that has finished its
// prompt samples one token and then decodes together with the rest in a single
// batched forward pass. The result is that concurrency raises total throughput
// instead of only dividing one stream's speed, which is the property continuous
// batching exists to provide.
//
// Cache memory is the constraint that decides how many requests can be in flight.
// The engine supports both layouts llama offers: a contiguous arena per sequence,
// which reserves a whole context on admission, and a paged cache drawing blocks
// from one shared pool, which charges a sequence only for the positions it uses.
// The second is what lets a queue be longer than the memory would otherwise
// allow.
//
// The package sits above llama and below the serving layer. It knows about
// sequences, slots, and fairness; it knows nothing about HTTP or JSON. That
// split is what lets the REPL, the CLI, and a future server share one path.
package engine
