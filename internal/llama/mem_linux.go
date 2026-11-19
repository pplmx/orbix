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
	return parseMemAvailable(string(data))
}

// parseMemAvailable extracts MemAvailable from the text of /proc/meminfo, or zero when the field is
// missing or unusable.
//
// Split out so the failure modes are testable: the file itself cannot be substituted in a unit test,
// and every one of these branches means "do not budget" rather than "budget zero bytes". A missing
// field, an empty one, a non-numeric one and a zero all have to reach the caller as the same 0, since
// a wrong guess here sizes a pool the process may not be able to fault in.
func parseMemAvailable(text string) int64 {
	for line := range strings.SplitSeq(text, "\n") {
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
