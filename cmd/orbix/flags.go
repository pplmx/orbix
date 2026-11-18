package main

import (
	"flag"
	"fmt"
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
func maxTokensFlag(fs *flag.FlagSet, def int) *int {
	return fs.Int("max-tokens", def, "cap on generated tokens, not counting the prompt")
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
// one-shot generation has no paged cache to size, but the worker count decides its speed --
// measured on the fixture, one worker gave 63.6 tok/s on a single stream and 128 gave 248.7,
// which is the difference between "this is slow" and "this is a usable demo".
func workersFlag(fs *flag.FlagSet) *int {
	return fs.Int("workers", 0,
		"kernel goroutines per batched pass; 0 means one per CPU. Useful work is bounded by "+
			"the outputs a pass can split, which for the vocabulary projection is its width "+
			"rather than the batch size -- measured on a 128-core host, 8 workers give about "+
			"two thirds of what 32 do, and beyond 32 the difference is inside the noise")
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
func (c cacheFlags) validate(prefix string) error {
	if *c.blocks < 0 || *c.blockSize < 0 {
		return fmt.Errorf("%s: --blocks and --block-size must not be negative", prefix)
	}
	if c.workers != nil && *c.workers < 0 {
		return fmt.Errorf("%s: --workers must not be negative", prefix)
	}
	return nil
}
