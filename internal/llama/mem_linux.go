//go:build linux

package llama

import (
	"os"
	"strconv"
	"strings"
)

// availableMemory reports how many bytes the kernel says it can hand out without swapping, or
// zero when that cannot be determined.
//
// MemAvailable is the kernel's own estimate -- it accounts for reclaimable page cache and for the
// low watermarks it keeps free -- which makes it a better answer than MemFree for "could this
// allocation succeed". Reading it is a few microseconds against a load that takes hundreds of
// milliseconds.
func availableMemory() int64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for line := range strings.SplitSeq(string(data), "\n") {
		rest, ok := strings.CutPrefix(line, "MemAvailable:")
		if !ok {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			return 0
		}
		// The value is in kibibytes.
		kb, err := strconv.ParseInt(fields[0], 10, 64)
		if err != nil || kb <= 0 {
			return 0
		}
		return kb * 1024
	}
	return 0
}
