package main

import (
	"flag"
	"strings"
	"testing"
)

// newCacheFlags builds a cacheFlags with the pointers cache.validate dereferences.
//
// A zero cacheFlags panics in validate -- its fields are pointers that cmdBench fills from flag
// values -- so every case that reaches the validation needs them present, even when the values
// are the defaults.
func newCacheFlags(blockSize, blocks int) cacheFlags {
	return cacheFlags{blockSize: &blockSize, blocks: &blocks}
}

// flagsFor builds the visit function validateBenchFlags needs, from a set of flag names.
//
// It mirrors what fs.Visit does -- report the flags that were actually set -- without a real
// FlagSet, because the check under test is about which names appear rather than about parsing.
func flagsFor(names ...string) func(func(*flag.Flag)) {
	return func(visit func(*flag.Flag)) {
		for _, n := range names {
			visit(&flag.Flag{Name: n})
		}
	}
}

// TestValidateBenchFlagsRefusesWhatTheSuiteCannotHonour covers the check that a flag the suite
// will not read is refused rather than accepted and ignored.
//
// Measured before the check existed: `--suite --policy bogus` ran the whole table and exited 0
// with the invalid value never looked at, and `--suite --concurrency 32` printed a table of eight
// scenarios measured at other concurrencies with nothing to say so. Both are a benchmark whose
// report does not describe what was measured.
func TestValidateBenchFlagsRefusesWhatTheSuiteCannotHonour(t *testing.T) {
	v := benchFlags{
		concurrency: 1, maxTokens: 8, suite: true,
		cache: newCacheFlags(0, 0),
		visit: flagsFor("suite", "concurrency", "policy"),
	}
	err := validateBenchFlags(v)
	if err == nil {
		t.Fatal("--suite with --concurrency and --policy was accepted")
	}
	// The message has to name the flags: an operator needs to know which one to drop, and a
	// generic "unused flags" leaves them guessing among the ones they passed.
	for _, want := range []string{"--concurrency", "--policy"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not name %s: %v", want, err)
		}
	}
}

// TestValidateBenchFlagsAcceptsWhatTheSuiteUses is the other half: the flags the suite does read
// must pass, or the check would refuse every legitimate invocation.
func TestValidateBenchFlagsAcceptsWhatTheSuiteUses(t *testing.T) {
	v := benchFlags{
		concurrency: 1, maxTokens: 8, suite: true,
		cache: newCacheFlags(0, 0),
		visit: flagsFor("suite", "model", "max-tokens", "prompt", "workers"),
	}
	if err := validateBenchFlags(v); err != nil {
		t.Errorf("the suite's own flags were refused: %v", err)
	}
}

// TestValidateBenchFlagsWithoutSuiteIgnoresTheSet covers that the check only applies to --suite.
//
// Without it every flag is read, so passing --concurrency and --policy is the normal case and
// must not be refused.
func TestValidateBenchFlagsWithoutSuiteIgnoresTheSet(t *testing.T) {
	v := benchFlags{
		concurrency: 4, maxTokens: 8, suite: false,
		cache: newCacheFlags(0, 0),
		visit: flagsFor("concurrency", "policy", "requests"),
	}
	if err := validateBenchFlags(v); err != nil {
		t.Errorf("a non-suite run was refused for using its own flags: %v", err)
	}
}

// TestValidateBenchFlagsRefusesNegativeRequestCount pins the value that has a documented meaning.
//
// Zero means "one request per concurrency slot", and the code reaches it with "total <= 0", so a
// negative value used to be silently the same as zero. In a benchmark the request count is the
// sample size: a mistyped one is not a smaller measurement but a different one.
func TestValidateBenchFlagsRefusesNegativeRequestCount(t *testing.T) {
	v := benchFlags{concurrency: 1, maxTokens: 8, requests: -5, cache: newCacheFlags(0, 0),
		visit: flagsFor()}
	err := validateBenchFlags(v)
	if err == nil {
		t.Fatal("--requests -5 was accepted")
	}
	if !strings.Contains(err.Error(), "-5") && !strings.Contains(err.Error(), "negative") {
		t.Errorf("the refusal does not say the value was negative: %v", err)
	}
}

// TestValidateBenchFlagsRefusesPrefixCacheWithoutAPool keeps the pair that only works together.
//
// --prefix-cache without --blocks used to run on the contiguous cache and then dereference the
// pool that --blocks would have created, so the run measured everything and panicked on its last
// print rather than reporting.
func TestValidateBenchFlagsRefusesPrefixCacheWithoutAPool(t *testing.T) {
	v := benchFlags{
		concurrency: 1, maxTokens: 8, prefixCache: true,
		// Both zero: a non-zero block size with no count is refused earlier, by a different
		// check, so this case has to reach the prefix-cache rule the way an operator would.
		cache: newCacheFlags(0, 0),
		visit: flagsFor("prefix-cache"),
	}
	err := validateBenchFlags(v)
	if err == nil {
		t.Fatal("--prefix-cache without --blocks was accepted")
	}
	// Both ways out, because either is a real fix and the operator should not have to guess.
	for _, want := range []string{"--blocks", "--prefix-cache"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not name %s: %v", want, err)
		}
	}
}

// TestValidateBenchFlagsRefusesAZeroBudget covers the one flag check in this function that had
// no test.
//
// The message is worth asserting in full because its reason is the one that differs from serve's
// and chat's: their zero-token requests are observable as empty replies, while bench's would
// produce a throughput figure of 0.0 from a run that generated nothing. A benchmark reporting 0.0
// looks like a slow server rather than like a request to measure nothing, which is why the value
// is refused instead of measured.
func TestValidateBenchFlagsRefusesAZeroBudget(t *testing.T) {
	v := benchFlags{concurrency: 1, maxTokens: 0, cache: newCacheFlags(0, 0), visit: flagsFor()}
	err := validateBenchFlags(v)
	if err == nil {
		t.Fatal("--max-tokens 0 was accepted")
	}
	for _, want := range []string{"--max-tokens", "at least 1", "0.0 tok/s"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not say %q: %v", want, err)
		}
	}
	// A negative value takes the same path, since the check is < 1 rather than == 0. Asserted
	// separately because a change to <= 0 would let -1 through and the zero case alone would
	// not notice.
	v.maxTokens = -1
	if err := validateBenchFlags(v); err == nil {
		t.Error("--max-tokens -1 was accepted")
	}
}
