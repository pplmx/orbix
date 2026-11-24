// Package arch holds no code. It holds the rules about which packages may import which.
//
// The dependency direction in this repository is a sentence in AGENTS.md and a table in
// docs/architecture.md, and a sentence does not fail a build. It has been broken before: four
// files in internal/server imported internal/llama to name a sampler and a log-probability
// alternative, which put a KV-cache package on the HTTP layer's import path. That was found by
// reading, not by a gate.
//
// This package is the gate. It parses every internal package's imports and fails on an edge the
// documented layering does not allow, so the rule holds without anyone re-reading the prose.
//
// It imports nothing and is never imported. A package with no non-test files still gets its tests
// run by `go test ./...`, which is all this needs.
package arch
