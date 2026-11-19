package main

import (
	"bytes"
	"errors"
	"reflect"
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
		{"tekken", "implemented"},
		// Every name llama.cpp maps to a family is implemented now, so the warning path is
		// covered in the tokenize package against the branch itself rather than through a
		// real name here. What is left to distinguish is a name that is not a family.
		{"kimi-k2", "implemented"},
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
		// "implemented" is a substring of "NOT implemented", so the assertion above cannot
		// tell the two apart: emitting the warning for every family passed this test. The
		// warning text is checked for directly instead.
		if c.want == "implemented" && strings.Contains(out, "NOT implemented") {
			t.Errorf("pre=%q: reported as implemented but the output warns it is not:\n%s", c.pre, out)
		}
	}
}

// TestJSONValueKeepsTheDeclaredWidth covers the type projection behind `model inspect --json`.
//
// The switch is on the GGUF type rather than on the field that happens to be populated,
// because the type is what the file records: a uint64 holding 5 is still an unsigned integer,
// and a script comparing against a Go int should see the width the file declares. Three of the
// eight type cases were unexercised -- uint64, int64 and float64 -- and dropping any of them
// passed the cmd suite, because every fixture and checkpoint the tests use carries only
// uint32/int32/float32 metadata.
//
// Dropping one does not fail loudly: the value falls to the default, which renders the summary
// form as a STRING. A JSON consumer then gets "1099511627776" instead of the number, and a
// comparison against a bound either misparses or silently types differently.
func TestJSONValueKeepsTheDeclaredWidth(t *testing.T) {
	cases := []struct {
		name string
		in   gguf.Value
		want any
	}{
		{"uint8", gguf.Value{Type: gguf.TypeUint8, Uint: 200}, uint64(200)},
		{"uint16", gguf.Value{Type: gguf.TypeUint16, Uint: 65535}, uint64(65535)},
		{"uint32", gguf.Value{Type: gguf.TypeUint32, Uint: 1 << 31}, uint64(1 << 31)},
		// Above int64's range, so a narrower handling would show up as a wrong number rather
		// than merely a different width.
		{"uint64", gguf.Value{Type: gguf.TypeUint64, Uint: 1 << 63}, uint64(1 << 63)},
		{"int8", gguf.Value{Type: gguf.TypeInt8, Int: -128}, int64(-128)},
		{"int16", gguf.Value{Type: gguf.TypeInt16, Int: -32768}, int64(-32768)},
		{"int32", gguf.Value{Type: gguf.TypeInt32, Int: -1 << 31}, int64(-1 << 31)},
		{"int64", gguf.Value{Type: gguf.TypeInt64, Int: -1 << 62}, int64(-1 << 62)},
		// The value a float32 actually holds: the reader widens the stored bits and keeps
		// the full float64 expansion, so a projection that widened it unchanged would emit
		// 9.999999747378752e-06 for a file that says 1e-5. FormatFloat narrows to 32 bits
		// before formatting, which is why the expected value is the round number.
		{"float32", gguf.Value{Type: gguf.TypeFloat32, Float: float64(float32(1e-5))}, 1e-5},
		{"float64", gguf.Value{Type: gguf.TypeFloat64, Float: 1e-300}, 1e-300},
		{"bool", gguf.Value{Type: gguf.TypeBool, Bool: true}, true},
		{"string", gguf.Value{Type: gguf.TypeString, Str: "hi"}, "hi"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := jsonValue(c.in)
			// The dynamic type matters as much as the value: that is what the width of the
			// JSON number follows from.
			if got != c.want {
				t.Errorf("jsonValue(%v) = %#v, want %#v", c.in.Type, got, c.want)
			}
			if wantT, gotT := reflect.TypeOf(c.want), reflect.TypeOf(got); wantT != gotT {
				t.Errorf("jsonValue(%v) returned %v, want %v", c.in.Type, gotT, wantT)
			}
		})
	}

	// The default keeps an array as its summary text, so a script can still see that a key
	// holds an array rather than getting a silently empty value.
	got := jsonValue(gguf.Value{Type: gguf.TypeArray, ElemType: gguf.TypeInt32})
	if _, ok := got.(string); !ok {
		t.Errorf("an array projected to %T, want the summary string", got)
	}
}

// TestErrWriterRemembersTheFirstFailure covers the "check once instead of at every call site"
// contract that errWriter's comment describes.
//
// Two behaviours matter and neither was tested. After a write fails, every later write must
// return that same error rather than trying the underlying writer again -- a sequence of prints
// that kept calling a broken writer would report the failure N times, or (worse, if the writer
// recovered) not at all. And a successful write must clear nothing: the first error is the one
// reported, because it is the one that explains the truncated output.
func TestErrWriterRemembersTheFirstFailure(t *testing.T) {
	first := errors.New("first failure")
	second := errors.New("second failure")

	t.Run("a failure is remembered and not retried", func(t *testing.T) {
		inner := &countingWriter{err: first}
		ew := &errWriter{w: inner}

		if _, err := ew.Write([]byte("a")); !errors.Is(err, first) {
			t.Fatalf("first write returned %v, want the underlying error", err)
		}
		calls := inner.calls
		for i := range 3 {
			_, err := ew.Write([]byte("b"))
			if !errors.Is(err, first) {
				t.Fatalf("write %d after a failure returned %v, want the remembered error", i+2, err)
			}
		}
		// The stored error is what a caller checks once at the end, and the inner writer was
		// not asked again: retrying a broken writer is what the memo exists to prevent.
		if ew.err != first {
			t.Errorf("errWriter.err = %v, want the first error", ew.err)
		}
		if inner.calls != calls {
			t.Errorf("the underlying writer was called %d more times after it failed", inner.calls-calls)
		}
	})

	t.Run("the first error wins", func(t *testing.T) {
		// A writer that fails differently each time: only the first may be reported.
		inner := &countingWriter{err: first}
		ew := &errWriter{w: inner}
		if _, err := ew.Write(nil); !errors.Is(err, first) {
			t.Fatalf("first write returned %v", err)
		}
		inner.err = second
		if _, err := ew.Write(nil); !errors.Is(err, first) {
			t.Errorf("after a later failure errWriter reported %v, want the first one", err)
		}
	})

	t.Run("success passes through", func(t *testing.T) {
		var buf bytes.Buffer
		ew := &errWriter{w: &buf}
		for _, s := range []string{"a", "b", "c"} {
			if _, err := ew.Write([]byte(s)); err != nil {
				t.Fatalf("Write(%q) = %v", s, err)
			}
		}
		if got := buf.String(); got != "abc" {
			t.Errorf("wrote %q, want abc", got)
		}
		if ew.err != nil {
			t.Errorf("a successful run left err = %v", ew.err)
		}
	})
}

// countingWriter fails with err and counts how many times it was asked to write.
type countingWriter struct {
	err   error
	calls int
}

func (w *countingWriter) Write(p []byte) (int, error) {
	w.calls++
	return 0, w.err
}

// TestModelSubcommandDispatch covers the three ways `orbix model` can be called and the three
// spellings of its help.
//
// Every alias has to work: an operator who types `orbix model help` after a failed first
// attempt must not be told the subcommand is unknown. Narrowing the case to just `-h` passed the
// cmd suite, because nothing called cmdModel at all -- the inspect tests go straight to the
// function below it.
func TestModelSubcommandDispatch(t *testing.T) {
	t.Run("no subcommand names what to type", func(t *testing.T) {
		err := cmdModel(nil)
		if err == nil {
			t.Fatal("cmdModel accepted an empty argument list")
		}
		if !strings.Contains(err.Error(), "inspect") {
			t.Errorf("error %q does not name the subcommand to use", err)
		}
	})

	t.Run("an unknown subcommand is named", func(t *testing.T) {
		err := cmdModel([]string{"inspekt"})
		if err == nil {
			t.Fatal("cmdModel accepted an unknown subcommand")
		}
		if !strings.Contains(err.Error(), "inspekt") {
			t.Errorf("error %q does not quote what was typed", err)
		}
		if !strings.Contains(err.Error(), "inspect") {
			t.Errorf("error %q does not name the subcommand to use", err)
		}
	})

	t.Run("every help spelling prints the usage", func(t *testing.T) {
		for _, spelling := range []string{"help", "--help", "-h"} {
			// cmdModel prints to os.Stdout, so this capture does not see it; what is asserted
			// is that the call succeeds rather than returning "unknown subcommand".
			if err := cmdModel([]string{spelling}); err != nil {
				t.Errorf("model %s: %v", spelling, err)
			}
		}
	})

	t.Run("a flag after the path is reported as ordering", func(t *testing.T) {
		// Go's flag package stops at the first non-flag, so `inspect path --json` parses no
		// flags and counts --json as a second path. "got 2" would not say that.
		err := cmdModelInspect([]string{"unused.gguf", "--json"})
		if err == nil {
			t.Fatal("model inspect accepted a flag after the path")
		}
		if !strings.Contains(err.Error(), "flags must come before") {
			t.Errorf("error %q does not explain the ordering", err)
		}
	})
}
