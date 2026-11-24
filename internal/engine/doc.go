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
// One goroutine drives the work. Submit queues a request and returns a channel; nothing
// runs until that goroutine calls Pump, and Pump must never be called concurrently. The
// reason is not a lock the engine declines to take: the model underneath owns one set of
// scratch buffers, so two rounds at once would write the same slices and produce output
// that is neither round's. Callers keep Pump on one goroutine and call Submit from
// wherever -- internal/server does exactly that.
//
// A caller that stops reading is given up on rather than waited for. Each sequence has a
// 64-event buffer; when it fills, the send waits two seconds for it to drain and then
// retires the sequence, freeing its slot. Without that bound one stalled reader would
// hold a slot and stall the shared pump for everyone, which is the failure that made the
// bound necessary. The consequence for a consumer is a real one and is written down in
// docs/usage.md: drain as you go, because a stream nobody reads is abandoned rather than
// paused.
//
// The package sits above llama and below the serving layer. It knows about
// sequences, slots, and fairness; it knows nothing about HTTP or JSON. That
// split is what lets the REPL, the CLI, the benchmark and internal/server share one
// path -- the server is the third caller, not a hypothetical one.
//
// # What the serving layer is allowed to name
//
// Three types here exist so that the package below does not leak upward. Sampler is one
// method, so llama's greedy and temperature samplers satisfy it without llama being named;
// Alternative and TokenBias are the log-probability and bias shapes a request carries. They
// moved here from llama after four files in internal/server imported that package purely to
// name them, which put a KV-cache package on the HTTP layer's import path for a vocabulary it
// did not need. internal/server/imports_test.go holds that edge shut.
//
// ErrPoolExhausted is the same idea for errors: the storage layer's "block pool exhausted"
// is translated at this boundary into a sentinel the scheduler owns, because whether it is
// temporary or permanent is a scheduling question. A caller can test for it with errors.Is
// and decide whether a Retry-After is worth honouring, without importing the cache.
//
// # Reading a round
//
// Drive is the loop most callers want and the one the CLI uses: it drains each sequence's
// channel on its own goroutine while pumping, then joins. Writing that by hand is four lines
// and easy to get wrong in the same way every time -- reading before pumping fills the
// buffers and stalls, which is what made the benchmark report a 64-token generation as 64
// tokens regardless of the budget. internal/server does not use it, because it needs to poll
// its own context and back off between rounds; the loop is spelled out there with the
// polling inline.
package engine
