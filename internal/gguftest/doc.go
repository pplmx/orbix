// Package gguftest builds small but complete GGUF files for tests.
//
// The engine, server, and CLI tests need a model they can actually run: attention needs
// a cache, generation needs an end-of-sequence token, and a tokenizer needs a vocabulary.
// Most of those tests were gated on the MODEL environment variable, which meant they were
// skipped wherever a checkpoint was not on disk -- in CI, all of them. A checkpoint is not
// needed for any of those assertions; what is needed is a model small enough to construct in
// a test and complete enough to run.
//
// No counts are quoted here. The figures that were (74 skipped of 320 tests when this package
// was written) move with every checkpoint-gated test added or removed, and nothing can pin a
// number that lives in prose; `go test ./... -v | grep -c '^--- SKIP'` is the answer for
// whatever tree is being read.
//
// The files this package produces are real GGUFs, parsed by the same reader a checkpoint
// goes through. They are deliberately tiny -- a few kilobytes, two layers, two attention
// heads -- because the assertions are about behaviour, not about capacity: a batched
// forward pass must match a serial one whatever the layer count is.
//
// This is not a test file, so any package's tests can import it. The builder it replaces
// lives inside internal/gguf's own tests, where only that package could reach it, which
// is why every other package grew its own copy or a MODEL gate.
//
// The vocabulary is a handful of characters rather than a trained tokenizer. That is
// enough for the properties these tests check -- round trips, special tokens, byte
// fallbacks -- and it keeps the fixture readable: a test that fails can be debugged by
// reading ten strings instead of a 151936-entry vocabulary.
package gguftest
