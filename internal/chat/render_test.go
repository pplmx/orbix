package chat

import (
	"strings"
	"testing"
)

// TestErrorMessagesCarryOneChatPrefix guards against a wrapped error saying "chat:" twice.
//
// There are two layers that both prefix: the evaluator writes "chat: undefined variable %q" and
// Render wraps whatever comes out with its own context. When that context also started with
// "chat:", the message read "chat: render: chat: undefined variable "x"" -- the same package name
// twice with nothing between them, which reads like a bug in the message rather than in the
// template.
//
// The fix is that the wrapper adds context only. This test pins it at the place a template author
// would see it: parse and render of a few broken templates, counting prefixes rather than
// matching whole strings, so the wording can change without the test having to.
func TestErrorMessagesCarryOneChatPrefix(t *testing.T) {
	cases := []struct{ name, src string }{
		{"undefined variable", "{{ x.y.z }}"},
		{"parse: early end", "{{ 1 + }}"},
		{"parse: for without in", "{% for %}"},
		{"unknown filter", "{{ x | nosuchfilter }}"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tpl, err := Parse(c.src)
			if err == nil {
				_, err = tpl.Render([]Message{{Role: "user", Content: "hi"}}, Inputs{})
			}
			if err == nil {
				t.Fatalf("%q produced no error", c.src)
			}
			if n := strings.Count(err.Error(), "chat:"); n != 1 {
				t.Errorf("%q gives %q, which names the package %d times; the wrapper should add "+
					"context, not a second prefix", c.src, err, n)
			}
		})
	}
}
