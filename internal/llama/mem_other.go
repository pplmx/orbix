//go:build !linux

package llama

// availableMemory returns zero on platforms without a cheap source for it, which disables the
// allocation check rather than guessing. The check exists to turn a corrupt header into an error
// instead of an hour of swapping; a wrong estimate would turn a legitimate model into a refusal,
// which is the worse failure.
func availableMemory() int64 { return 0 }
