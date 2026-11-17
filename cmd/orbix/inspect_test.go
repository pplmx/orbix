package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
)

// TestInspectReportsPreTokenizer checks the line a reader needs before judging a model's
// output quality: whether this build reproduces the file's pre-tokenizer.
//
// The failure it guards against is silence. An unimplemented family splits text as one
// chunk, which produces tokens the model was never trained on; the model still answers, so
// nothing looks broken. Printing the state is the only place a user can find that out.
func TestInspectReportsPreTokenizer(t *testing.T) {
	cases := []struct {
		pre  string
		want string
	}{
		{"qwen2", "implemented"},
		{"default", "implemented"},
		{"deepseek-v3", "implemented"},
		// Still unimplemented: the warning path needs a family that really is missing.
		{"tekken", "NOT implemented"},
		{"qwen-2", "NOT a name this package knows"},
	}
	for _, c := range cases {
		path := gguftest.Model{
			Vocab:          []string{"<unk>", "<s>", "</s>", "\u2581", "a"},
			Merges:         []string{"a a"},
			TokenizerModel: "gpt2",
			NoBytes:        true,
			Pre:            c.pre,
			ContextLen:     64,
		}.Write(t, t.TempDir())
		f, err := gguf.Open(path)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		var buf bytes.Buffer
		if err := inspect(&buf, f, path, inspectOptions{}); err != nil {
			t.Fatalf("inspect: %v", err)
		}
		out := buf.String()
		if !strings.Contains(out, "Pre-tokenizer") {
			t.Fatalf("pre=%q: output has no Pre-tokenizer line:\n%s", c.pre, out)
		}
		if !strings.Contains(out, c.want) {
			t.Errorf("pre=%q: output does not say %q:\n%s", c.pre, c.want, out)
		}
	}
}
