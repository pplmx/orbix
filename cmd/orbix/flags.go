package main

import (
	"flag"
	"fmt"
	"math"
	"os"
)

// The subcommands share a small set of flags, and before this file each one spelled
// them out: --model in four places, --max-tokens in four, the cache flags in three. The
// copies drifted -- one command explained --max-tokens as "tokens to generate" and
// another as "maximum tokens" -- so a reader comparing two commands' help output could
// not tell whether the difference was meaningful. Registering each flag once is what
// keeps the meaning in one place.

// modelFlag registers --model and returns the pointer to resolve it through.
func modelFlag(fs *flag.FlagSet) *string {
	return fs.String("model", "", "path to a GGUF model file (required, or set ORBIX_MODEL)")
}

// resolveModel fills in a model path from the environment when the flag was not given.
//
// The environment is read here rather than through a configuration library because there
// is exactly one variable and no configuration file: a library for merging flag, file,
// and environment sources would earn its place the moment a second source exists.
func resolveModel(fs *flag.FlagSet, model *string) error {
	if *model != "" {
		return nil
	}
	if v := os.Getenv("ORBIX_MODEL"); v != "" {
		*model = v
		return nil
	}
	return fmt.Errorf("%s: --model is required (or set ORBIX_MODEL)", fs.Name())
}

// maxTokensFlag registers --max-tokens with the one wording every command shares.
//
// The commands do not all accept the same values, and the wording says which is which rather
// than leaving them to be discovered to disagree: run accepts 0, which prefills and generates
// nothing; bench refuses it, because a measurement of zero generated tokens is not a
// measurement; serve and chat refuse it too, and for a different reason in each -- serve's is a
// default for requests that omit the field, and server.Config reads anything below 1 as 256, so
// 0 there would configure 256 under a flag that says 0; a REPL turn that generates nothing is an
// empty reply. The wording names the two that accept a value the others refuse; the refusals
// carry their own explanations in the error, which is where a reader can act on them.
//
// Neither of the refusals is about what a request can carry. A request body's own
// `max_tokens: 0` is a pointer on the wire and is served as a zero-token reply, which is a
// separate rule from this flag.
func maxTokensFlag(fs *flag.FlagSet, def int) *int {
	return fs.Int("max-tokens", def,
		"cap on generated tokens, not counting the prompt; run accepts 0 (prefill only) and "+
			"bench, serve and chat need 1 or more")
}

// cacheFlags are the paged-cache settings several commands share.
type cacheFlags struct {
	blockSize *int
	blocks    *int
	workers   *int
}

// workersFlag registers --workers with the one wording every command shares.
//
// It is separate from registerCacheFlags because `run` wants it without the cache flags: a
// one-shot generation has no paged cache to size, but the worker count decides its speed.
//
// The figures here used to be "one worker gave 63.6 tok/s on a single stream and 128 gave
// 248.7" -- a 3.9x claim that stopped being true as the fixture and the kernels changed.
// Re-measured with `orbix run --model .cache/stress.gguf --workers N --max-tokens 64 --stats`,
// three runs each on the same 128-core host: one worker 707.9, 709.0, 705.1; 128 workers
// 943.3, 916.6, 984.3. That is 1.34x, not 3.9x.
//
// 128 workers buy little because the fixture cannot use them. `parts()` returns
// min(workers, work/16384), and this fixture's vocabulary projection is 363 x 256 = 92928
// multiply-adds, so the split stops at 5 whatever the flag says -- `--workers 128` runs five.
// That is also why the measurements above put 8, 32 and 128 within noise of each other.
//
// The help text used to carry those four numbers, prefixed with "on the checkpoint from just
// fixture". A user reading `--help` has no fixture and cannot reproduce them, and the sentence
// never said what to do with the figure -- so it now states the behaviour instead: the split is
// bounded by the pass's smallest dimension, and this flag cannot help a request that is waiting
// for a slot. The measurements stay here, where the command that produced them is named.
//
// It must not name another flag, and my first version of the replacement did: it said a request
// "queued behind --max-active" waits the same, but this text is shared by run and chat, which
// have no --max-active at all -- run has no admission limit, and chat's is implied by --blocks.
// A reader of `orbix run -h` was told about a flag that command does not accept. The wording says
// "waiting for a slot" now, which is true of every command that registers this.
//
// This paragraph has now been wrong three times. It first said 128 workers were 3.9x faster than
// one; that was true of an earlier fixture and stopped being true as the fixture changed. I
// replaced it with an explanation that the split was not the bound, computing 32000 x 256 --
// 32000 is the vocabulary of the benchmark in internal/kernel, not of this fixture. The original
// claim's explanation was right and I deleted it while correcting its number. Then the help text
// carried 214/240/246 tok/s without saying which model or which command, and re-measuring the
// fixture it named gave 703/937/934/996 -- four times higher, and rising from one worker instead
// of flat. The numbers below now come with the command that produces them.
func workersFlag(fs *flag.FlagSet) *int {
	return fs.Int("workers", 0,
		"kernel goroutines per batched pass; 0 means one per CPU. Past a point the flag stops "+
			"helping: a pass can only be split as far as its smallest dimension allows, so a "+
			"small model does not use a large number, and a request that is waiting for a slot "+
			"does not run faster for it either")
}

// samplingFlags holds the three knobs that decide how a token is drawn.
type samplingFlags struct {
	temp float64
	topK int
	topP float64
}

// registerSamplingFlags registers --temp, --top-k and --top-p.
//
// run and chat each registered these three by hand, with wording that had to stay identical and a
// validation block that had to stay equivalent. They were identical when this was written, which
// is the state that makes the duplication invisible: nothing fails until one of them is edited.
// The validation was the part worth sharing, because it is not a range check -- NaN and Inf pass
// every comparison, and a NaN temperature reached the sampler, made the sum NaN, failed every
// "r < acc" and returned the last candidate, the least likely token in the set.
func registerSamplingFlags(fs *flag.FlagSet, s *samplingFlags) {
	// *Var rather than the returning form, so the fields stay plain values: validate's tests build
	// a struct literal, and a caller reading the result after Parse needs no dereference.
	fs.Float64Var(&s.temp, "temp", 0, "sampling temperature; 0 means greedy")
	fs.IntVar(&s.topK, "top-k", 40, "top-k sampling cutoff, 0 disables it. Only applies with --temp > 0")
	fs.Float64Var(&s.topP, "top-p", 0, "nucleus sampling cutoff in (0,1], 0 disables it. Only applies with --temp > 0")
}

// validate refuses sampling values with no reading, naming the command that received them.
//
// The command name is the only thing that differs between callers, so it is a parameter rather
// than a reason to restate the four checks.
func (s samplingFlags) validate(cmd string) error {
	// Sampling is switched on by "temp > 0" and top-k applies at "topK > 0", so a negative value
	// is the same as zero: the run goes greedy without saying so.
	if math.IsNaN(s.temp) || math.IsInf(s.temp, 0) || s.temp < 0 {
		return fmt.Errorf("%s: --temp must be a finite, non-negative number (0 means greedy)", cmd)
	}
	if s.topK < 0 {
		return fmt.Errorf("%s: --top-k must not be negative (0 disables it)", cmd)
	}
	// A probability outside [0, 1] is refused rather than clamped. Zero and one both mean "no
	// limit" to the sampler and the flag documents 0 as the value that disables the cutoff, so
	// what is refused is a value with no reading at all: clamping -0.1 up to 0 would silently
	// turn a typo into "off", and lowering 1.5 would apply a cutoff nobody asked for.
	if math.IsNaN(s.topP) || s.topP < 0 || s.topP > 1 {
		return fmt.Errorf("%s: --top-p must be between 0 and 1 (0 disables it)", cmd)
	}
	return nil
}

// registerCacheFlags registers the paged-cache flags, and the worker flag when asked.
func registerCacheFlags(fs *flag.FlagSet, workers bool) cacheFlags {
	var c cacheFlags
	c.blockSize = fs.Int("block-size", 0, "KV cache block size in positions; 0 keeps the contiguous cache")
	c.blocks = fs.Int("blocks", 0, "KV cache blocks shared by all requests; implies a paged cache")
	if workers {
		c.workers = workersFlag(fs)
	}
	return c
}

// validate rejects negative cache settings, which would otherwise reach the pool
// constructor as a confusing error far from the flag that caused it.
//
// autoBlocks says the caller sizes a pool itself when --blocks is unset, which chat does from
// the model's own context because a REPL wants paging plus prefix reuse by default. For it a
// block size alone is meaningful; for serve and bench it is not, because the engine keeps its
// contiguous cache and the number never reaches anything.
func (c cacheFlags) validate(prefix string, autoBlocks bool) error {
	// Named separately rather than as one "the cache flags" sentence: a caller who typed
	// `--blocks -1` and is told that "--blocks and --block-size must not be negative" has to
	// re-read their own command line to work out which of the two they got wrong.
	if *c.blocks < 0 {
		return fmt.Errorf("%s: --blocks must not be negative", prefix)
	}
	if *c.blockSize < 0 {
		return fmt.Errorf("%s: --block-size must not be negative", prefix)
	}
	// --block-size only sets how the pool's blocks are divided; without --blocks there is no
	// pool to divide. Accepting it alone leaves the contiguous cache in place, which on a
	// 128k-context model reserves 8 GB per accepted request -- and the operator who passed a
	// block size reads the resulting "kv cache: contiguous" line as the size they asked for.
	//
	// Checked here rather than in each command so a command that registers these flags cannot
	// forget it: chat did, and silently ignored a block size nobody could see was unused.
	if !autoBlocks && *c.blockSize > 0 && *c.blocks == 0 {
		return fmt.Errorf("%s: --block-size needs --blocks; without a pool there are no "+
			"blocks to size, and each request still reserves a whole context", prefix)
	}
	if c.workers != nil && *c.workers < 0 {
		return fmt.Errorf("%s: --workers must not be negative", prefix)
	}
	return nil
}
