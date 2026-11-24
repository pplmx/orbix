package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

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

	// A non-finite float has to survive to the encoder, and this is where it does so. A GGUF
	// float is any bit pattern and the reader accepts whatever the bits say, so a checkpoint
	// carrying a NaN is legal input; encoding/json refuses NaN and both infinities because
	// JSON's number grammar has no token for them, and a value returned here unchanged makes
	// the whole document fail rather than one field. Measured before the fix: a file with
	// test.nan made `model inspect --json` print "orbix: json: unsupported value: NaN" and
	// nothing else, so no metadata could be read at all.
	//
	// The string spelling is asserted rather than merely "not a float", because "NaN", "+Inf"
	// and "-Inf" are what strconv produces and what a reader can recognise. A null would
	// decode as absent and a zero as a real number; both lose the fact the file records.
	nonFinite := []struct {
		name string
		in   gguf.Value
		want string
	}{
		{"NaN", gguf.Value{Type: gguf.TypeFloat64, Float: math.NaN()}, "NaN"},
		{"+Inf", gguf.Value{Type: gguf.TypeFloat64, Float: math.Inf(1)}, "+Inf"},
		{"-Inf", gguf.Value{Type: gguf.TypeFloat64, Float: math.Inf(-1)}, "-Inf"},
		{"float32 NaN", gguf.Value{Type: gguf.TypeFloat32, Float: math.NaN()}, "NaN"},
	}
	for _, c := range nonFinite {
		t.Run(c.name, func(t *testing.T) {
			if got := jsonValue(c.in); got != c.want {
				t.Errorf("jsonValue(%v) = %#v, want the string %q", c.in.Type, got, c.want)
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

	// The other four commands fall back to ORBIX_MODEL, this one does not, and a caller who
	// exports it runs `model inspect` bare expecting the same thing. Saying the variable is
	// not used answers the question the message would otherwise leave open.
	t.Run("a set ORBIX_MODEL is called out as not used", func(t *testing.T) {
		t.Setenv("ORBIX_MODEL", "/some/checkpoint.gguf")
		err := cmdModelInspect(nil)
		if err == nil {
			t.Fatal("model inspect accepted no path")
		}
		if !strings.Contains(err.Error(), "ORBIX_MODEL is not used here") {
			t.Errorf("error %q does not say the variable is ignored", err)
		}
		// The variable must not be read as the path either: that would inspect a file the
		// caller did not name, which is the reason it is not consulted.
		if strings.Contains(err.Error(), "/some/checkpoint.gguf") {
			t.Errorf("error %q used ORBIX_MODEL as a path", err)
		}
	})

	// With the variable unset the message stays the plain one: there is no expectation to
	// correct, so naming the variable would only add a line to read.
	t.Run("without the variable the message is unchanged", func(t *testing.T) {
		t.Setenv("ORBIX_MODEL", "")
		err := cmdModelInspect(nil)
		if err == nil {
			t.Fatal("model inspect accepted no path")
		}
		if strings.Contains(err.Error(), "ORBIX_MODEL") {
			t.Errorf("error %q mentions ORBIX_MODEL with none set", err)
		}
	})
}

// TestInspectSurvivesAFileWithNoMetadata covers the two degraded paths of the inspect report.
//
// A GGUF file may carry no metadata at all: the format allows it, and a converter writing a bare
// tensor directory produces one. The report still has to describe what it can rather than stopping,
// because the reader's question -- "what is this file?" -- is answered by the tensor list and the
// version even when the architecture and tokenizer are absent.
//
// Both lines were uncovered: the architecture printed "(unknown)" and the tokenizer printed
// "(unusable: ...)". They matter because the alternative is an error that hides the tensor
// directory, which is the only part of such a file that is informative.
func TestInspectSurvivesAFileWithNoMetadata(t *testing.T) {
	// GGUF v3 with no key/value pairs and no tensors, which is the smallest valid file.
	var b []byte
	b = append(b, []byte("GGUF")...)
	var u32 [4]byte
	binary.LittleEndian.PutUint32(u32[:], 3)
	b = append(b, u32[:]...)
	var u64 [8]byte
	binary.LittleEndian.PutUint64(u64[:], 0) // no tensors
	b = append(b, u64[:]...)
	binary.LittleEndian.PutUint64(u64[:], 0) // no metadata
	b = append(b, u64[:]...)

	path := filepath.Join(t.TempDir(), "bare.gguf")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("writing the file: %v", err)
	}

	var runErr error
	out := captureStdout(t, func() { runErr = run([]string{"model", "inspect", path}) })
	if runErr != nil {
		t.Fatalf("inspect refused a file with no metadata: %v", runErr)
	}
	// The writer is a tabwriter, so the fields are separated by runs of spaces rather than by
	// tabs; the assertions match the label and value without assuming a separator.
	for _, want := range []string{
		"Architecture",
		"(unknown)",
		"(unusable:",
		"GGUF version",
		"Tensors",
		// The name falls back to the file's base name, which is the only identifier such a
		// file has.
		"bare",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report does not contain %q:\n%s", want, out)
		}
	}

	// A file that names an unknown architecture still reports it rather than the fallback, so the
	// two cases are distinguishable to a reader.
	named := gguftest.Model{Arch: "gemma", NoEOS: true}.Write(t, t.TempDir())
	out = captureStdout(t, func() { runErr = run([]string{"model", "inspect", named}) })
	if runErr != nil {
		t.Fatalf("inspect refused an unknown architecture: %v", runErr)
	}
	if !strings.Contains(out, "gemma") {
		t.Errorf("an unknown-but-named architecture did not print its name:\n%s", out)
	}
}

// TestInspectMarksAnUnsizeableTypePartial covers the "(partial)" suffix on a row whose byte count
// could not be computed.
//
// The suffix exists so a reader is not told a confident number for a type this build cannot size:
// the count of tensors is still exact, the size is not. Nothing tested it -- removing the suffix and
// forcing the flag false both passed.
//
// The file is hand-built because the fixture writer only emits F32 tensors, and the condition is a
// type the package does not know.
func TestInspectMarksAnUnsizeableTypePartial(t *testing.T) {
	var b []byte
	b = append(b, []byte("GGUF")...)
	var u32 [4]byte
	var u64 [8]byte
	binary.LittleEndian.PutUint32(u32[:], 3)
	b = append(b, u32[:]...)
	binary.LittleEndian.PutUint64(u64[:], 1) // one tensor
	b = append(b, u64[:]...)
	binary.LittleEndian.PutUint64(u64[:], 0) // no metadata
	b = append(b, u64[:]...)
	// The tensor: name, one dimension, an unknown type id, offset.
	var nameLen [8]byte
	binary.LittleEndian.PutUint64(nameLen[:], 1)
	b = append(b, nameLen[:]...)
	b = append(b, 'w')
	binary.LittleEndian.PutUint32(u32[:], 1)
	b = append(b, u32[:]...)
	binary.LittleEndian.PutUint64(u64[:], 4) // one dimension: 4
	b = append(b, u64[:]...)
	binary.LittleEndian.PutUint32(u32[:], 9999) // not a type this build knows
	b = append(b, u32[:]...)
	binary.LittleEndian.PutUint64(u64[:], 0) // offset
	b = append(b, u64[:]...)

	path := filepath.Join(t.TempDir(), "odd.gguf")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("writing the file: %v", err)
	}

	out := captureStdout(t, func() {
		if err := run([]string{"model", "inspect", path}); err != nil {
			t.Errorf("inspect refused a file with an unknown tensor type: %v", err)
		}
	})
	if !strings.Contains(out, "(partial)") {
		t.Errorf("a row whose size could not be computed is not marked partial:\n%s", out)
	}
	// And the tensor itself still appears, so the reader knows how many there are.
	if !strings.Contains(out, "w") {
		t.Errorf("the tensor is missing from the report:\n%s", out)
	}
}

// TestInspectJSONSurvivesNonFiniteFloats is the end-to-end half of the NaN case.
//
// The unit test above pins what jsonValue returns for one value. This drives the command with
// a real file, because the failure it guards against was not in the projection alone: encoding
// a document that held a NaN made the whole write fail, so the command printed one line and
// exited non-zero, and a reader could get nothing out of a file whose other 17 keys were
// perfectly ordinary.
//
// The assertion is that the output parses as JSON at all, then that the values are where they
// belong. A test that only checked for the absence of an error would pass on an empty document.
func TestInspectJSONSurvivesNonFiniteFloats(t *testing.T) {
	var b []byte
	var u32 [4]byte
	var u64 [8]byte

	putStr := func(v string) []byte {
		binary.LittleEndian.PutUint64(u64[:], uint64(len(v)))
		return append(append([]byte(nil), u64[:]...), v...)
	}
	putKey := func(key string, typ uint32) {
		b = append(b, putStr(key)...)
		binary.LittleEndian.PutUint32(u32[:], typ)
		b = append(b, u32[:]...)
	}
	putF64 := func(key string, bits uint64) {
		putKey(key, 12) // float64
		binary.LittleEndian.PutUint64(u64[:], bits)
		b = append(b, u64[:]...)
	}

	b = append(b, []byte("GGUF")...)
	binary.LittleEndian.PutUint32(u32[:], 3)
	b = append(b, u32[:]...)
	binary.LittleEndian.PutUint64(u64[:], 0) // no tensors
	b = append(b, u64[:]...)

	// Four entries: the architecture, an ordinary float to prove the document is intact, and
	// the three non-finite values JSON cannot spell.
	binary.LittleEndian.PutUint64(u64[:], 5)
	b = append(b, u64[:]...)
	putKey("general.architecture", 8) // string
	b = append(b, putStr("llama")...)
	putF64("test.finite", math.Float64bits(1.5))
	putF64("test.nan", math.Float64bits(math.NaN()))
	putF64("test.inf", math.Float64bits(math.Inf(1)))
	putF64("test.neg_inf", math.Float64bits(math.Inf(-1)))

	path := filepath.Join(t.TempDir(), "nonfinite.gguf")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("writing the file: %v", err)
	}

	var runErr error
	out := captureStdout(t, func() { runErr = run([]string{"model", "inspect", "--json", path}) })
	if runErr != nil {
		t.Fatalf("inspect --json refused a file with a NaN float: %v", runErr)
	}

	var doc struct {
		Metadata map[string]any `json:"metadata"`
	}
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}
	// The finite value has to still be a number: turning every float into a string would fix
	// the crash by giving up on the common case.
	if got := doc.Metadata["test.finite"]; got != 1.5 {
		t.Errorf("test.finite = %#v, want the number 1.5", got)
	}
	for key, want := range map[string]string{
		"test.nan":     "NaN",
		"test.inf":     "+Inf",
		"test.neg_inf": "-Inf",
	} {
		if got := doc.Metadata[key]; got != want {
			t.Errorf("%s = %#v, want the string %q", key, got, want)
		}
	}
	// And the rest of the document survives, which is the point: one unusual value must not
	// cost the reader every other key.
	if got := doc.Metadata["general.architecture"]; got != "llama" {
		t.Errorf("general.architecture = %#v, want llama", got)
	}
}

// TestTruncateForDisplayKeepsUTF8Valid covers the helper the two tables share.
//
// A tensor name or a metadata string is arbitrary UTF-8, so a byte-counted cut can land in the
// middle of a multi-byte character. The earlier version of this logic was written for metadata
// values alone and left tensor names untruncated entirely, which meant a 300-character name
// pushed every other column of the tensors table off screen -- measured on a hand-built file,
// the type and size columns were no longer visible at a normal terminal width.
//
// The property is utf8.ValidString on the result, not a particular length: any boundary that
// keeps the output valid is acceptable, and asserting the exact cut would make a cosmetic
// change look like a regression.
func TestTruncateForDisplayKeepsUTF8Valid(t *testing.T) {
	cases := []struct {
		name string
		in   string
		max  int
	}{
		{"short ASCII", "blk.0.attn_q.weight", 60},
		{"exactly at the limit", strings.Repeat("a", 60), 60},
		{"one over", strings.Repeat("a", 61), 60},
		{"long ASCII", strings.Repeat("very_long_name_", 20), 60},
		// A cut landing inside a multi-byte rune: every 3-byte character means the 57-byte
		// boundary is a continuation byte.
		{"CJK at the boundary", strings.Repeat("中", 30), 60},
		{"emoji at the boundary", strings.Repeat("🙂", 30), 60},
		{"mixed", "abc" + strings.Repeat("中", 25), 60},
		{"wide max", strings.Repeat("x", 10), 100},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := truncateForDisplay(c.in, c.max)
			if !utf8.ValidString(got) {
				t.Errorf("truncateForDisplay(%q, %d) = %q, which is not valid UTF-8", c.in, c.max, got)
			}
			if len(got) > c.max {
				t.Errorf("result is %d bytes, over the %d budget: %q", len(got), c.max, got)
			}
			if len(c.in) <= c.max {
				if got != c.in {
					t.Errorf("a string inside the budget was changed: %q -> %q", c.in, got)
				}
				return
			}
			if !strings.HasSuffix(got, "...") {
				t.Errorf("a truncated string %q does not end in an ellipsis", got)
			}
			// The kept prefix has to be a prefix: truncation must not rewrite the text.
			if !strings.HasPrefix(c.in, strings.TrimSuffix(got, "...")) {
				t.Errorf("%q is not a prefix of the input", got)
			}
		})
	}
}

// TestTruncateForDisplayHandlesBudgetsUnderTheEllipsis covers the widths the test above never
// used: every one of its cases passes a budget around 60, which is what the three call sites
// use today.
//
// That made a panic invisible. `keep := max - 3` goes negative when max is under the
// ellipsis's own length, and s[:keep] then fails with "slice bounds out of range [:-1]" --
// measured at max 2, 1 and 0. No caller can reach it now, but the function's contract is
// "shorten s to max bytes", and a helper that panics for part of its stated domain is a trap
// for whoever raises or lowers a column width next.
//
// Two behaviours are pinned rather than one, because they differ and both are deliberate: a
// budget between 1 and 3 returns the ellipsis, which is wider than asked and is the only way
// to say "this was cut"; a budget of 0 returns nothing, because a zero-width column has no
// room for the marker either.
func TestTruncateForDisplayHandlesBudgetsUnderTheEllipsis(t *testing.T) {
	const long = "abcdef"
	for _, c := range []struct {
		max  int
		want string
	}{
		{5, "ab..."},
		{4, "a..."},
		{3, "..."},
		{2, "..."},
		{1, "..."},
		{0, ""},
		{-1, ""},
	} {
		got := truncateForDisplay(long, c.max)
		if got != c.want {
			t.Errorf("truncateForDisplay(%q, %d) = %q, want %q", long, c.max, got, c.want)
		}
	}
	// A multi-byte input at a sub-ellipsis budget must not panic before it reaches the guard.
	if got := truncateForDisplay("中文字符", 1); got != "..." {
		t.Errorf("truncateForDisplay(CJK, 1) = %q, want \"...\"", got)
	}
}

// TestInspectTruncatesLongTensorNamesOnce covers the table the helper was extended for.
//
// The assertion is about the rendered row, not about the helper: the point of the change is
// that the columns after the name stay on screen, and only a test that looks at the printed
// line can tell whether the call was wired in. A tensor named with 300 characters must have
// its type and size visible after it, and --json must still carry the whole name, because a
// script looking the tensor up needs the real one.
func TestInspectTruncatesLongTensorNamesOnce(t *testing.T) {
	long := strings.Repeat("very_long_name_", 20)
	// A minimal file built here rather than through gguftest, which fixes the tensor names it
	// writes: the whole point is a name longer than the display budget.
	var b []byte
	u32 := func(v uint32) { var t [4]byte; binary.LittleEndian.PutUint32(t[:], v); b = append(b, t[:]...) }
	u64 := func(v uint64) { var t [8]byte; binary.LittleEndian.PutUint64(t[:], v); b = append(b, t[:]...) }
	str := func(s string) { u64(uint64(len(s))); b = append(b, s...) }
	b = append(b, []byte("GGUF")...)
	u32(3)
	u64(1) // one tensor
	u64(1) // one metadata entry
	str("general.architecture")
	u32(8) // string
	str("llama")
	str(long)
	u32(1) // one dimension
	u64(4)
	u32(0) // F32
	u64(0) // offset
	for len(b)%32 != 0 {
		b = append(b, 0)
	}
	b = append(b, make([]byte, 16)...)
	data := b
	path := filepath.Join(t.TempDir(), "long.gguf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	var runErr error
	out := captureStdout(t, func() { runErr = run([]string{"model", "inspect", path}) })
	if runErr != nil {
		t.Fatalf("inspect: %v", runErr)
	}

	var row string
	for _, line := range strings.Split(out, "\n") {
		if strings.Contains(line, "very_long_name_") {
			row = line
		}
	}
	if row == "" {
		t.Fatalf("the tensor row is missing from:\n%s", out)
	}
	if strings.Contains(row, long) {
		t.Errorf("the row carries the whole %d-character name, so the table is unreadable:\n%s",
			len(long), row)
	}
	if !strings.Contains(row, "...") {
		t.Errorf("the truncated name has no ellipsis:\n%s", row)
	}
	// The columns that follow have to still be there.
	for _, want := range []string{"F32", "4", "16 B"} {
		if !strings.Contains(row, want) {
			t.Errorf("the row lost %q after the name:\n%s", want, row)
		}
	}

	// And the machine-readable form keeps it whole.
	jsonOut := captureStdout(t, func() { runErr = run([]string{"model", "inspect", "--json", path}) })
	if runErr != nil {
		t.Fatalf("inspect --json: %v", runErr)
	}
	var doc struct {
		Tensors []struct {
			Name string `json:"name"`
		} `json:"tensors"`
	}
	if err := json.Unmarshal([]byte(jsonOut), &doc); err != nil {
		t.Fatalf("decoding: %v", err)
	}
	if len(doc.Tensors) != 1 {
		t.Fatalf("JSON has %d tensors, want 1", len(doc.Tensors))
	}
	if doc.Tensors[0].Name != long {
		t.Errorf("JSON name is %d characters, want the full %d",
			len(doc.Tensors[0].Name), len(long))
	}
}

// TestInspectCannotBeMadeToPrintExtraRows covers what an untrusted model file can do to the
// human-readable table.
//
// Every string in a GGUF is attacker-controlled: general.name, general.architecture, the
// metadata values and the tensor names. tabwriter ends a row at a newline and starts a column
// at a tab, so a name containing both lets the file print statistics about itself that the tool
// never measured. Measured before the sanitizer existed, a fixture whose general.name was
// "Evil\nArchitecture\tfake" produced a line reading "Architecture  fake" above the real
// "Architecture (unknown)" row.
//
// The assertion counts lines rather than pattern-matching, because the failure is structural:
// an injected row is invisible to any check that only looks for the strings the injector
// planted. The two-directory fixture gives a fixed line count to compare against.
func TestInspectCannotBeMadeToPrintExtraRows(t *testing.T) {
	clean := &gguf.File{Metadata: map[string]gguf.Value{
		"general.name":         {Type: gguf.TypeString, Str: "MyModel"},
		"general.architecture": {Type: gguf.TypeString, Str: "llama"},
	}}
	var cleanBuf bytes.Buffer
	if err := writeHuman(&cleanBuf, clean, "/x.gguf"); err != nil {
		t.Fatalf("writeHuman: %v", err)
	}
	wantLines := bytes.Count(cleanBuf.Bytes(), []byte("\n"))

	for _, c := range []struct {
		name  string
		value string
	}{
		{"a newline in general.name", "Evil\nArchitecture\tfake"},
		{"a carriage return", "Evil\rArchitecture\tfake"},
		{"only whitespace controls", "\n\t\r"},
		{"a lone newline", "\n"},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := &gguf.File{Metadata: map[string]gguf.Value{
				"general.name":         {Type: gguf.TypeString, Str: c.value},
				"general.architecture": {Type: gguf.TypeString, Str: "llama"},
			}}
			var buf bytes.Buffer
			if err := writeHuman(&buf, f, "/x.gguf"); err != nil {
				t.Fatalf("writeHuman: %v", err)
			}
			if got := bytes.Count(buf.Bytes(), []byte("\n")); got != wantLines {
				t.Errorf("a name of %q produced %d lines, want %d: the file injected "+
					"%d of its own", c.value, got, wantLines, got-wantLines)
			}
			// The planted text must not appear as a label at the start of a line, which is
			// what an injected row would look like.
			for _, line := range strings.Split(buf.String(), "\n") {
				if strings.HasPrefix(strings.TrimSpace(line), "fake") {
					t.Errorf("the file started a line with its own text: %q", line)
				}
			}
		})
	}

	// The value is still shown, with the separators replaced rather than dropped: hiding it
	// would leave a reader unable to tell a sanitized name from a truncated one.
	t.Run("the text survives", func(t *testing.T) {
		f := &gguf.File{Metadata: map[string]gguf.Value{
			"general.name":         {Type: gguf.TypeString, Str: "Evil\nArchitecture\tfake"},
			"general.architecture": {Type: gguf.TypeString, Str: "llama"},
		}}
		var buf bytes.Buffer
		if err := writeHuman(&buf, f, "/x.gguf"); err != nil {
			t.Fatalf("writeHuman: %v", err)
		}
		if !strings.Contains(buf.String(), "Evil Architecture fake") {
			t.Errorf("the sanitized name is missing from:\n%s", buf.String())
		}
	})
}
