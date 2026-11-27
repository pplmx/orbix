//go:build race

package chat

// raceBuild is true when the test binary was built with -race, which `just test-race` arranges
// by passing -tags race (the go tool does not set a tag for -race on its own).
//
// Tests that assert on wall-clock time read this. Race instrumentation adds several times the
// work per memory access, so a budget calibrated on an ordinary build measures the
// instrumentation rather than the algorithm -- TestParseScalesLinearlyInTagCount failed CI at
// 76ms against a 60ms budget that the same code clears in 9-12ms un-instrumented.
//
// This is not a licence to skip what a test checks under -race. Race builds are where the
// concurrency assertions matter most, and every one of those still runs; only the clock is
// unreadable there.
const raceBuild = true
