// Package engine ties the model together with request scheduling: it owns the
// loaded weights, hands out per-sequence KV caches, and turns a stream of
// prompts into a stream of tokens.
//
// The package sits above llama and below the serving layer. It knows about
// sequences, slots, and fairness; it knows nothing about HTTP or JSON. That
// split is what lets the REPL, the CLI, and a future server share one path.
package engine
