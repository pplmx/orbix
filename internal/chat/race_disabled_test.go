//go:build !race

package chat

// raceBuild is false for ordinary test binaries. See race_enabled_test.go for why the timing
// tests need to know.
const raceBuild = false
