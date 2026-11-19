//go:build linux

package llama

import "testing"

// TestParseMemAvailable covers every way the field can be unusable.
//
// Each of these branches means "do not budget" -- the caller treats zero as unknown and sizes the pool
// without a memory check. They were unreachable from a test while the parsing lived inside
// availableMemory, which reads the real file: the failures need a file that is missing, truncated or
// malformed, and none of those can be arranged against /proc.
func TestParseMemAvailable(t *testing.T) {
	cases := []struct {
		name string
		text string
		want int64
	}{
		{
			name: "a normal file",
			text: "MemTotal:       16384000 kB\nMemFree:         1234567 kB\n" +
				"MemAvailable:    8000000 kB\nBuffers:          100000 kB\n",
			want: 8000000 * 1024,
		},
		{
			// The field is not always first, and the parse must not stop at MemTotal.
			name: "the field after others",
			text: "MemTotal: 1 kB\nBuffers: 2 kB\nMemAvailable: 4096 kB\n",
			want: 4096 * 1024,
		},
		{name: "no such field", text: "MemTotal: 1000 kB\n", want: 0},
		{name: "an empty file", text: "", want: 0},
		{name: "an empty value", text: "MemAvailable:\n", want: 0},
		{name: "a non-numeric value", text: "MemAvailable: plenty kB\n", want: 0},
		// Zero means the kernel could hand out nothing, which is indistinguishable from unknown for
		// the caller and must not be reported as a budget of zero bytes.
		{name: "a zero value", text: "MemAvailable: 0 kB\n", want: 0},
		// A negative count is nonsense from the parser's side and must not become a negative budget.
		{name: "a negative value", text: "MemAvailable: -5 kB\n", want: 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := parseMemAvailable(c.text); got != c.want {
				t.Errorf("parseMemAvailable(%q) = %d, want %d", c.text, got, c.want)
			}
		})
	}
}
