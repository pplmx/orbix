package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"

	"github.com/pplmx/orbix/internal/llama"
)

// The CLI tests drive run() directly, which keeps them free of subprocesses
// and lets them assert on stdout content rather than on exit codes alone.

func TestRunNoArgsPrintsUsage(t *testing.T) {
	out := captureStdout(t, func() {
		if err := run(nil); err != nil {
			t.Fatalf("run(nil) = %v", err)
		}
	})
	for _, want := range []string{"Usage:", "model inspect", "version"} {
		if !strings.Contains(out, want) {
			t.Errorf("usage output missing %q:\n%s", want, out)
		}
	}
}

// TestCommandHelpIsNotAFailure covers the exit status of `orbix <command> -h`.
//
// The usage text tells the reader to run exactly that to see a command's flags, and the flag
// package prints them and returns flag.ErrHelp to say "nothing went wrong, this was a request
// for help". Reporting it as an error made every one of those commands exit 1 with an extra
// "orbix: flag: help requested" line, while `orbix help` exited 0 -- two documented paths to
// the same information, disagreeing on whether asking for it succeeded.
//
// The assertion is on report's status rather than on a subprocess, because main calls
// os.Exit: the mapping is what has to be right, and testing it directly also keeps the test
// from depending on an exec.
func TestCommandHelpIsNotAFailure(t *testing.T) {
	// Every command that has its own flag set. `version` and `help` are absent because they
	// handle -h themselves and never reach the flag package; they are covered below.
	commands := [][]string{
		{"run", "-h"},
		{"chat", "-h"},
		{"bench", "-h"},
		{"serve", "-h"},
		{"model", "inspect", "-h"},
	}
	for _, args := range commands {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			var stderr bytes.Buffer
			// The flag package writes the help text itself; discard it so the test output
			// stays readable, and keep stderr only to prove nothing extra was added.
			old := os.Stderr
			devnull, err := os.Open(os.DevNull)
			if err != nil {
				t.Fatal(err)
			}
			os.Stderr = devnull
			err = run(args)
			os.Stderr = old
			_ = devnull.Close()

			if got := report(err, &stderr); got != 0 {
				t.Errorf("report(run(%v)) = %d, want 0 for a help request", args, got)
			}
			if strings.Contains(stderr.String(), "help requested") {
				t.Errorf("help request reported as an error: %q", stderr.String())
			}
		})
	}

	// The half that has to keep failing: a genuine error still exits non-zero and still says
	// what went wrong, so the help case above cannot pass by mapping everything to 0.
	var stderr bytes.Buffer
	if got := report(errors.New("boom"), &stderr); got != 1 {
		t.Errorf("report(err) = %d, want 1", got)
	}
	if !strings.Contains(stderr.String(), "boom") {
		t.Errorf("error message not written to stderr: %q", stderr.String())
	}
	if got := report(nil, &stderr); got != 0 {
		t.Errorf("report(nil) = %d, want 0", got)
	}
}

func TestRunUnknownCommand(t *testing.T) {
	err := run([]string{"nope"})
	if err == nil {
		t.Fatal("run(unknown) succeeded, want error")
	}
	if !strings.Contains(err.Error(), "unknown command") {
		t.Errorf("error = %v, want it to mention an unknown command", err)
	}
}

func TestModelInspectRequiresPath(t *testing.T) {
	if err := run([]string{"model", "inspect"}); err == nil {
		t.Fatal("model inspect with no path succeeded, want error")
	}
	if err := run([]string{"model"}); err == nil {
		t.Fatal("model with no subcommand succeeded, want error")
	}
	if err := run([]string{"model", "bogus"}); err == nil {
		t.Fatal("model bogus succeeded, want error")
	}
}

// TestModelInspectNamesAFlagAfterThePath covers the mistake Go's flag package invites.
//
// flag stops parsing at the first non-flag argument, so `inspect ./m.gguf --json` parses no
// flags and counts --json as a second path. The old message was "want exactly one file path,
// got 2", which is true and does not help: the caller passed one path and believes the flag was
// honoured.
//
// Checked by running both orders against the same fixture: `inspect --json ./m.gguf` prints
// JSON, `inspect ./m.gguf --json` errored. The message now names the flag and where it has to
// go.
func TestModelInspectNamesAFlagAfterThePath(t *testing.T) {
	err := run([]string{"model", "inspect", "model.gguf", "--json"})
	if err == nil {
		t.Fatal("a flag after the path was accepted, want an error")
	}
	msg := err.Error()
	if !strings.Contains(msg, "before the path") {
		t.Errorf("error %q does not say where the flag belongs", msg)
	}
	if !strings.Contains(msg, "--json") {
		t.Errorf("error %q does not name the flag that was misplaced", msg)
	}

	// A genuinely extra path must still report the count rather than blaming a flag.
	err = run([]string{"model", "inspect", "a.gguf", "b.gguf"})
	if err == nil {
		t.Fatal("two paths were accepted, want an error")
	}
	if strings.Contains(err.Error(), "before the path") {
		t.Errorf("two paths produced the flag-order message: %q", err)
	}
}

func TestModelInspectMissingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.gguf")
	err := run([]string{"model", "inspect", path})
	if err == nil {
		t.Fatal("inspect on a missing file succeeded, want error")
	}
}

func TestModelInspectHuman(t *testing.T) {
	path := writeTestModel(t)
	out := captureStdout(t, func() {
		if err := run([]string{"model", "inspect", path}); err != nil {
			t.Fatalf("inspect: %v", err)
		}
	})

	// The summary must name the architecture and count what it found. Column
	// padding is tabwriter's business, so match on the label and value as
	// separate fields rather than on exact spacing.
	for _, want := range []string{
		"Architecture",
		"llama",
		"GGUF version",
		"Metadata entries",
		"Tensors",
		"token_embd.weight",
		"Q4_0",
		"F32",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
	if !hasField(out, "Tensors", "2") {
		t.Errorf("output missing the tensor count 2:\n%s", out)
	}
	if !hasField(out, "Architecture", "llama") {
		t.Errorf("output missing the architecture:\n%s", out)
	}
	// Dimensions are printed outermost-first, which is not ggml's storage
	// order, so a [4 2] tensor reads as "2 x 4".
	if !strings.Contains(out, "2 x 4") {
		t.Errorf("output missing the transposed shape %q:\n%s", "2 x 4", out)
	}
}

func TestModelInspectJSON(t *testing.T) {
	path := writeTestModel(t)
	out := captureStdout(t, func() {
		if err := run([]string{"model", "inspect", "--json", path}); err != nil {
			t.Fatalf("inspect --json: %v", err)
		}
	})

	var got struct {
		Version       uint32         `json:"version"`
		Alignment     int64          `json:"alignment"`
		TensorDataLoc int64          `json:"tensor_data_offset"`
		Metadata      map[string]any `json:"metadata"`
		Tensors       []struct {
			Name   string   `json:"name"`
			Dims   []uint64 `json:"dims"`
			Type   string   `json:"type"`
			Offset uint64   `json:"offset"`
			Bytes  *int64   `json:"bytes"`
		} `json:"tensors"`
	}
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}
	if got.Version != 3 {
		t.Errorf("version = %d, want 3", got.Version)
	}
	if got.Alignment != 32 {
		t.Errorf("alignment = %d, want 32", got.Alignment)
	}
	if got.TensorDataLoc <= 0 || got.TensorDataLoc%32 != 0 {
		t.Errorf("tensor_data_offset = %d, want a positive multiple of 32", got.TensorDataLoc)
	}
	if len(got.Tensors) != 2 {
		t.Fatalf("tensors = %d, want 2", len(got.Tensors))
	}
	if got.Tensors[0].Name != "token_embd.weight" || got.Tensors[0].Type != "F32" {
		t.Errorf("tensor[0] = %+v", got.Tensors[0])
	}
	// A known size proves the byte accounting made it through the CLI.
	if got.Tensors[0].Bytes == nil || *got.Tensors[0].Bytes != 32 {
		t.Errorf("tensor[0].bytes = %v, want 32", got.Tensors[0].Bytes)
	}
	// The values keep their GGUF type. This is the assertion whose absence let the output ship
	// as a map of strings: a script reading a numeric field had to parse "128" back out, which
	// is the work a --json flag exists to remove.
	if v, ok := got.Metadata["general.architecture"]; !ok {
		t.Error("metadata missing general.architecture")
	} else if _, isStr := v.(string); !isStr {
		t.Errorf("a GGUF string decoded as %T, want string", v)
	}
	// An integer field is a JSON number, and the value is checked as well as the type:
	// encoding/json decodes any JSON number into float64, so a type check alone would pass
	// for a field whose string form happened to be numeric.
	if v, ok := got.Metadata["llama.block_count"]; !ok {
		t.Error("metadata missing llama.block_count")
	} else if f, isNum := v.(float64); !isNum {
		t.Errorf("llama.block_count decoded as %T (%v), want a number", v, v)
	} else if f != 1 {
		t.Errorf("llama.block_count = %v, want 1", f)
	}
	// A float field is a number at its own precision, not the widened float64 spelled out in
	// full: 1e-5 as a float32 is 9.999999747378752e-06, and printing that is noise the file
	// cannot justify.
	if v, ok := got.Metadata["llama.attention.layer_norm_rms_epsilon"]; !ok {
		t.Error("metadata missing llama.attention.layer_norm_rms_epsilon")
	} else if f, isNum := v.(float64); !isNum {
		t.Errorf("the epsilon decoded as %T (%v), want a number", v, v)
	} else if f != 1e-5 {
		t.Errorf("epsilon = %v, want 1e-5 (float32 precision, not the widened double)", f)
	}
	// An array stays a summary: a tokenizer carries three arrays of tens of thousands of
	// entries, and inlining them would bury the rest of the document.
	if v, ok := got.Metadata["tokenizer.ggml.tokens"]; ok {
		if _, isStr := v.(string); !isStr {
			t.Errorf("an array decoded as %T (%v), want the summary string", v, v)
		}
	}
}

// TestVersionJSONAndUnknownArgs covers the two things `version` got wrong.
//
// It read its arguments itself -- it only looked for -h -- so `version --json` printed the human
// report and exited zero. A build script piping that into a JSON parser got a parse error where
// the flag said success. The other commands refuse an unknown flag through the flag package;
// this one silently ignored it.
//
// The JSON exists because this is the command a script reads: the cross-build job checks that an
// artifact reports cgo:false, and parsing a two-column text report for that is worse than parsing
// a document.
// TestRunAcceptsWorkers covers the flag `run` was missing while every other command had it.
//
// run builds the model through llama.New directly rather than through the engine, so the worker
// flag the engine-based commands register never reached it: the one parameter that decides a
// single stream's speed could not be set. Measured on the fixture, 1 worker gave 63.5 tok/s and
// 32 gave 248.3 -- a command whose default reported "this is slow" with no way to ask for more.
// TestNegativeCacheFlagsNameTheOneGiven covers a message that was true and unhelpful.
//
// Both cache flags were checked together and reported as "must not be negative", so a caller
// who typed `--blocks -1` was told that --blocks and --block-size were both wrong. Working out
// which one they had given meant re-reading their own command line.
//
// The prefix comes from the caller, so the same check reports serve, chat and bench correctly;
// the test uses serve and asserts each message names only the flag that was negative.
func TestNegativeCacheFlagsNameTheOneGiven(t *testing.T) {
	path := chatModel(t)

	err := run([]string{"serve", "--model", path, "--blocks", "-1"})
	if err == nil {
		t.Fatal("serve --blocks -1 succeeded, want an error")
	}
	if !strings.Contains(err.Error(), "--blocks") {
		t.Errorf("error %q does not name --blocks", err)
	}
	if strings.Contains(err.Error(), "--block-size") {
		t.Errorf("error %q blames --block-size, which was not given", err)
	}

	err = run([]string{"serve", "--model", path, "--block-size", "-1"})
	if err == nil {
		t.Fatal("serve --block-size -1 succeeded, want an error")
	}
	if !strings.Contains(err.Error(), "--block-size") {
		t.Errorf("error %q does not name --block-size", err)
	}
	if strings.Contains(err.Error(), "--blocks must") {
		t.Errorf("error %q blames --blocks, which was not given", err)
	}
}

func TestRunAcceptsWorkers(t *testing.T) {
	path := chatModel(t)

	out := captureStdout(t, func() {
		if err := run([]string{"run", "--model", path, "--workers", "4",
			"--max-tokens", "2", "Once upon a time"}); err != nil {
			t.Fatalf("run --workers: %v", err)
		}
	})
	if strings.TrimSpace(out) == "" {
		t.Error("run --workers produced no output")
	}

	// A negative count is refused rather than reaching the kernel runner, which would divide
	// the work by a non-positive number.
	err := run([]string{"run", "--model", path, "--workers", "-1", "x"})
	if err == nil {
		t.Fatal("run --workers -1 succeeded, want an error")
	}
	if !strings.Contains(err.Error(), "workers") {
		t.Errorf("error %q does not name --workers", err)
	}
}

func TestVersionJSONAndUnknownArgs(t *testing.T) {
	out := captureStdout(t, func() {
		if err := run([]string{"version", "-json"}); err != nil {
			t.Fatalf("version -json: %v", err)
		}
	})
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("version -json did not produce JSON: %v\n%s", err, out)
	}
	for _, want := range []string{"version", "commit", "built", "go", "platform", "cgo"} {
		if _, ok := got[want]; !ok {
			t.Errorf("version -json is missing %q; got %v", want, got)
		}
	}
	if _, ok := got["cgo"].(bool); !ok {
		t.Errorf("cgo is %T, want bool: %v", got["cgo"], got["cgo"])
	}

	// An unknown argument must fail rather than print the report and exit zero.
	if err := run([]string{"version", "--bogus"}); err == nil {
		t.Error("version --bogus succeeded, want an error")
	} else if !strings.Contains(err.Error(), "--bogus") {
		t.Errorf("error %q does not name the argument", err)
	}
}

func TestVersionCommand(t *testing.T) {
	out := captureStdout(t, func() {
		if err := run([]string{"version"}); err != nil {
			t.Fatalf("version: %v", err)
		}
	})
	for _, want := range []string{"orbix ", "commit:", "go:", "platform:"} {
		if !strings.Contains(out, want) {
			t.Errorf("version output missing %q:\n%s", want, out)
		}
	}

	// Every field's value starts in the same column. This report is read by a person comparing two
	// builds, mostly to answer "is this binary the commit I think it is", and a value that starts
	// further left than its neighbours makes that scan slower for no reason. The check caught a real
	// case: "platform:" carried no padding while every other label did, and "CGO_ENABLED:" sat flush
	// against its value because the label is exactly as wide as the column was.
	var columns []int
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		if !strings.HasPrefix(line, "  ") {
			continue
		}
		// Where the value begins: the label is separated from it by a run of spaces after a colon.
		// Searching for ":" alone finds the hour inside a timestamp, and counting from the colon
		// without skipping the padding would measure the label's length rather than the column.
		colon := strings.Index(line, ":")
		if colon < 0 {
			continue
		}
		rest := line[colon+1:]
		trimmed := strings.TrimLeft(rest, " ")
		if trimmed == "" || len(trimmed) == len(rest) {
			// No padding: the value is flush against the colon, which is the failure this check is
			// for rather than a line to measure.
			columns = append(columns, -1)
			continue
		}
		columns = append(columns, len(line)-len(trimmed))
	}
	if len(columns) < 4 {
		t.Fatalf("found %d labelled fields, want at least 4:\n%s", len(columns), out)
	}
	for _, c := range columns[1:] {
		if c != columns[0] {
			t.Errorf("value column varies: %v (from %v)", columns, columns[0])
			break
		}
	}
}

func TestFormatBytes(t *testing.T) {
	tests := map[int64]string{
		0:               "0 B",
		512:             "512 B",
		1024:            "1.00 KiB",
		1024 * 1024:     "1.00 MiB",
		3 * 1024 * 1024: "3.00 MiB",
	}
	for in, want := range tests {
		if got := formatBytes(in); got != want {
			t.Errorf("formatBytes(%d) = %q, want %q", in, got, want)
		}
	}
}

func TestFormatShape(t *testing.T) {
	tests := []struct {
		dims []uint64
		want string
	}{
		{nil, "scalar"},
		{[]uint64{64}, "64"},
		{[]uint64{64, 4096}, "4096 x 64"},
		{[]uint64{8, 4, 2}, "2 x 4 x 8"},
	}
	for _, tc := range tests {
		if got := formatShape(tc.dims); got != tc.want {
			t.Errorf("formatShape(%v) = %q, want %q", tc.dims, got, tc.want)
		}
	}
}

// captureStdout runs fn with os.Stdout redirected and returns what it wrote.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stdout
	os.Stdout = w
	done := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		buf.ReadFrom(r)
		done <- buf.String()
	}()

	defer func() {
		os.Stdout = old
	}()
	fn()
	w.Close()
	os.Stdout = old
	return <-done
}

// hasField reports whether some line starts with label and has want as its
// whitespace-separated second field. Tabwriter output is aligned with runs of
// spaces whose width depends on neighbouring labels, so tests must not pin
// the exact gap.
func hasField(out, label, want string) bool {
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && fields[0] == label && fields[1] == want {
			return true
		}
	}
	return false
}

// TestNegativeSamplingFlagsAreRejected checks the sampling flags that have no sensible value
// outside their range.
//
// Both --temp and --top-k are tested with "> 0" to decide whether they apply, so a negative
// value means greedy decoding and no top-k cut -- the opposite of turning sampling down.
// --top-p is a probability and is out of range at both ends.
//
// The flags are checked before the model is opened, so this needs no checkpoint.
func TestNegativeSamplingFlagsAreRejected(t *testing.T) {
	cases := []struct {
		cmd  string
		args []string
		want string
	}{
		{"run", []string{"--model", "unused.gguf", "--temp", "-1", "hi"}, "--temp"},
		{"run", []string{"--model", "unused.gguf", "--top-k", "-1", "hi"}, "--top-k"},
		// --top-p is a probability, so both ends are meaningless: below 0 and above 1. It
		// is refused rather than clamped because 0 and 1 both read as "no limit" to some
		// callers and "only the best token" to others.
		{"run", []string{"--model", "unused.gguf", "--top-p", "-0.5", "hi"}, "--top-p"},
		{"run", []string{"--model", "unused.gguf", "--top-p", "1.5", "hi"}, "--top-p"},
		{"chat", []string{"--model", "unused.gguf", "--temp", "-1"}, "--temp"},
		{"chat", []string{"--model", "unused.gguf", "--top-k", "-1"}, "--top-k"},
		{"chat", []string{"--model", "unused.gguf", "--top-p", "-0.5"}, "--top-p"},
		{"chat", []string{"--model", "unused.gguf", "--top-p", "1.5"}, "--top-p"},
	}
	for _, c := range cases {
		t.Run(c.cmd+" "+c.want, func(t *testing.T) {
			var err error
			if c.cmd == "run" {
				err = cmdRun(c.args)
			} else {
				err = cmdChat(c.args)
			}
			if err == nil {
				t.Fatalf("%s accepted %v", c.cmd, c.args)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not name %s", err, c.want)
			}
			if strings.Contains(err.Error(), "unused.gguf") {
				t.Errorf("error %q came from opening the model, so the flag was not validated", err)
			}
		})
	}
}

// TestNegativeMaxTokensIsRejected checks the flag that produced a silent empty answer.
//
// The engine rejected a negative count, but only on the path through Submit; run reaches the
// model through Generate, which had no check, so --max-tokens -1 printed nothing and exited
// successfully. Both commands must refuse it rather than leave the user looking at blank
// output with no error.
func TestNegativeMaxTokensIsRejected(t *testing.T) {
	err := cmdRun([]string{"--model", "unused.gguf", "--max-tokens", "-1", "hi"})
	if err == nil {
		t.Fatal("run accepted --max-tokens -1")
	}
	// The message names the flag as the user typed it, which is what makes it findable.
	if !strings.Contains(err.Error(), "max-tokens") {
		t.Errorf("error %q does not name --max-tokens", err)
	}
}

// TestGenerateRejectsNegativeMaxTokens covers the library check that the command relies on.
func TestGenerateRejectsNegativeMaxTokens(t *testing.T) {
	// A nil model is enough: the check runs before anything touches the weights.
	var m *llama.Model
	if _, err := m.Generate(nil, []int32{1}, nil, llama.GenerateOptions{MaxTokens: -1}); err == nil {
		t.Fatal("Generate accepted a negative MaxTokens")
	} else if !strings.Contains(err.Error(), "negative") {
		t.Errorf("error %q does not say the count was negative", err)
	}
}

// TestEveryNumericFlagRefusesNegatives is a sweep rather than a list of cases chosen by
// reading the code.
//
// Eight flags on these commands turned a negative value into the default by comparing with
// "<= 0" or "> 0", so the server or the sampler carried on with something other than what was
// asked for. Each was found one at a time, by passing a wrong value and noticing nothing
// happened. Driving every numeric flag through the same wrong value is what turns that into a
// check: the next flag added to any of these commands either rejects negatives or fails here.
//
// The assertion is deliberately on the message naming the flag, because an error from opening
// the model would also be non-nil and would say nothing about the flag.
func TestEveryNumericFlagRefusesNegatives(t *testing.T) {
	cases := []struct {
		name string
		run  func() error
		flag string
	}{
		{"run --max-tokens", func() error {
			return cmdRun([]string{"--model", "unused.gguf", "--max-tokens", "-1", "hi"})
		}, "--max-tokens"},
		{"run --temp", func() error {
			return cmdRun([]string{"--model", "unused.gguf", "--temp", "-1", "hi"})
		}, "--temp"},
		{"run --top-k", func() error {
			return cmdRun([]string{"--model", "unused.gguf", "--top-k", "-1", "hi"})
		}, "--top-k"},
		{"run --ctx", func() error {
			return cmdRun([]string{"--model", "unused.gguf", "--ctx", "-1", "hi"})
		}, "--ctx"},
		{"chat --max-tokens", func() error {
			return cmdChat([]string{"--model", "unused.gguf", "--max-tokens", "-1"})
		}, "--max-tokens"},
		{"chat --temp", func() error {
			return cmdChat([]string{"--model", "unused.gguf", "--temp", "-1"})
		}, "--temp"},
		{"chat --top-k", func() error {
			return cmdChat([]string{"--model", "unused.gguf", "--top-k", "-1"})
		}, "--top-k"},
		{"chat --workers", func() error {
			return cmdChat([]string{"--model", "unused.gguf", "--workers", "-1"})
		}, "--workers"},
		{"serve --max-queued", func() error {
			return cmdServe([]string{"--model", "unused.gguf", "--max-queued", "-1"})
		}, "--max-queued"},
		{"serve --workers", func() error {
			return cmdServe([]string{"--model", "unused.gguf", "--workers", "-1"})
		}, "--workers"},
		{"bench --concurrency", func() error {
			return cmdBench([]string{"--model", "unused.gguf", "--concurrency", "-1"})
		}, "--concurrency"},
		{"bench --requests", func() error {
			return cmdBench([]string{"--model", "unused.gguf", "--requests", "-1"})
		}, "--requests"},
		{"bench --max-tokens", func() error {
			return cmdBench([]string{"--model", "unused.gguf", "--max-tokens", "-1"})
		}, "--max-tokens"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := c.run()
			if err == nil {
				t.Fatalf("%s accepted a negative value", c.name)
			}
			msg := err.Error()
			if !strings.Contains(msg, c.flag) {
				t.Errorf("error %q does not name %s", msg, c.flag)
			}
			if strings.Contains(msg, "unused.gguf") {
				t.Errorf("error %q came from opening the model, so the flag was not validated", msg)
			}
		})
	}
}

// TestRunRefusesContextPastTheTrainedWindow covers the one --ctx value that looks usable and is not.
//
// The flag's help says "0 uses the model's trained amount", which reads as though any other value is
// allowed. A larger one is not: the rope table is built from the trained context, so a bigger cache
// asks it for positions it does not hold and the decode loop fails mid-run with "position 128 outside
// the rope table" -- after the prompt has been processed and part of the answer printed. Extending a
// model past its window needs rope scaling, which this engine does not implement.
//
// Checked against a constructed model so the numbers are exact: the fixture's ContextLen is the
// window, and one token past it is the smallest request that must be refused.
func TestRunRefusesContextPastTheTrainedWindow(t *testing.T) {
	const window = 64
	path := gguftest.Model{NoEOS: true, ContextLen: window}.Write(t, t.TempDir())

	// Anything up to the trained window is usable, as long as the prompt fits -- "hi" encodes to
	// three tokens against this vocabulary, so the smaller cases leave room for it.
	for _, n := range []string{"8", fmt.Sprint(window)} {
		if err := cmdRun([]string{"--model", path, "--ctx", n, "--max-tokens", "1", "hi"}); err != nil {
			t.Errorf("--ctx %s was refused: %v", n, err)
		}
	}

	err := cmdRun([]string{"--model", path, "--ctx", fmt.Sprint(window + 1), "hi"})
	if err == nil {
		t.Fatal("--ctx past the trained window was accepted")
	}
	// The message has to say what the limit is and why it exists; "outside the rope table" from
	// inside the decode loop does neither.
	for _, want := range []string{"--ctx", fmt.Sprint(window + 1), "rope scaling"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// TestRunSeedOnlyMattersWithTemperature pins the condition the --seed help text states.
//
// A seed without --temp does nothing: greedy decoding is already deterministic, so every run
// prints the same text and the seed is unread. That is documented in the flag's help and in
// the server's request struct, but nothing tested it, and "seed=42 gives the same answer
// twice" reads like the seed working when it is the decoder ignoring it.
//
// The test drives the command rather than the sampler so it covers the wiring: a seed with
// no temperature must be byte-identical across runs, and with a temperature the same seed
// must still reproduce while different seeds diverge.
func TestRunSeedOnlyMattersWithTemperature(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())

	run := func(args ...string) string {
		t.Helper()
		return captureStdout(t, func() {
			base := []string{"--model", path, "--max-tokens", "12", "Once upon a time"}
			if err := cmdRun(append(args, base...)); err != nil {
				t.Fatalf("cmdRun(%v): %v", args, err)
			}
		})
	}

	// Without --temp: greedy, so the seed changes nothing and every run matches.
	greedyA := run("--seed", "42")
	greedyB := run("--seed", "99")
	if greedyA != greedyB {
		t.Errorf("greedy output differed across seeds:\n%q\n%q", greedyA, greedyB)
	}

	// With --temp: the same seed reproduces, and a different seed is expected to diverge.
	sampledA := run("--temp", "1.5", "--seed", "42")
	sampledB := run("--temp", "1.5", "--seed", "42")
	if sampledA != sampledB {
		t.Errorf("the same seed and temperature produced different text:\n%q\n%q", sampledA, sampledB)
	}
	// A different seed on random weights should differ; if it does not, the run is so short
	// or the distribution so flat that the test cannot tell, which is worth knowing rather
	// than asserting.
	if other := run("--temp", "1.5", "--seed", "7"); other == sampledA {
		t.Logf("seed 7 and seed 42 gave the same %d-token text; "+
			"this run cannot distinguish seeds, but reproducibility held", len(sampledA))
	}
}

// TestBenchReportsPrefixReuse covers the line that tells an operator whether turning on
// --prefix-cache did anything.
//
// Without it a one-off bench printed the same report whether sharing happened or not, so a run
// whose prompt was too short to fill a block looked identical to one where the cache was broken.
// --suite has always reported the number for its reuse pair; this is the single-run path.
//
// Gated on MODEL because the figures only exist once real weights have prefilled something.
func TestBenchReportsPrefixReuse(t *testing.T) {
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file")
	}
	// Long enough to span more than one default 16-position block, which is the condition for
	// anything to be shareable at all.
	longPrompt := strings.Repeat("word ", 24)

	cases := []struct {
		name string
		args []string
		want string
		why  string
	}{
		{
			name: "cache off prints no reuse line",
			args: []string{"--model", path, "--max-tokens", "2", "--concurrency", "2",
				"--requests", "4", "--prompt", longPrompt},
			want: "",
			why:  "a run that could not reuse anything should not claim a figure",
		},
		{
			name: "cache on with a shareable prompt prints a count",
			args: []string{"--model", path, "--max-tokens", "2", "--concurrency", "2",
				"--requests", "8", "--prefix-cache", "--blocks", "128", "--block-size", "16",
				"--prompt", longPrompt},
			want: "prompt positions",
			why:  "the requests share a prompt that spans whole blocks, so some of it is reused",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := captureStdout(t, func() {
				if err := cmdBench(c.args); err != nil {
					t.Fatalf("cmdBench: %v", err)
				}
			})
			got := false
			for _, line := range strings.Split(out, "\n") {
				if strings.HasPrefix(line, "reused:") {
					got = true
					if c.want == "" {
						t.Errorf("unexpected reuse line: %q", line)
					} else if !strings.Contains(out, c.want) {
						t.Errorf("reuse line %q does not contain %q", line, c.want)
					}
				}
			}
			if c.want != "" && !got {
				t.Errorf("no reuse line printed; want one containing %q (%s)\noutput:\n%s",
					c.want, c.why, out)
			}
			if c.want == "" && got {
				t.Errorf("a reuse line was printed for a run whose cache was off (%s)", c.why)
			}
		})
	}
}

// TestBenchRejectsCacheFlagsWithoutABlockPool covers the two cache settings that bench
// accepted and then either ignored or crashed on.
//
// --prefix-cache without --blocks used to complete the whole measurement and then panic on
// the run's last print: the reuse note reads the pool's block size to explain why nothing was
// shared, and without --blocks there is no pool. A panic after a full benchmark is worse than
// an early error in both directions -- the numbers look like a finished result and the exit
// status is a crash -- and the underlying setting was wrong even when the print was reached.
//
// --block-size without --blocks was the quieter version of the same mistake: the run stayed on
// the contiguous cache while the report printed a contiguous footprint, so the size that was
// passed appeared nowhere and nothing said it had been dropped.
//
// Both are rejected before the model is opened, so this needs no checkpoint. The last case is
// the accepting half: --blocks with --prefix-cache must still reach the model, or the check
// could pass by refusing every combination.
func TestBenchRejectsCacheFlagsWithoutABlockPool(t *testing.T) {
	rejected := []struct {
		name string
		args []string
		want string
		why  string
	}{
		{
			name: "prefix cache without a pool",
			args: []string{"--prefix-cache"},
			want: "--blocks",
			why:  "the reuse note dereferences the pool, so the run panicked after measuring",
		},
		{
			name: "block size without a pool",
			args: []string{"--block-size", "32"},
			want: "--blocks",
			why:  "the contiguous cache stayed in place and the block size appeared nowhere",
		},
	}
	for _, c := range rejected {
		t.Run(c.name, func(t *testing.T) {
			args := append([]string{"--model", "unused.gguf", "--max-tokens", "1"}, c.args...)
			err := cmdBench(args)
			if err == nil {
				t.Fatalf("cmdBench accepted %v; %s", c.args, c.why)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not mention %q", err, c.want)
			}
			if strings.Contains(err.Error(), "unused.gguf") {
				t.Errorf("error %q came from opening the model, so the flag was not validated", err)
			}
		})
	}

	// --blocks alone is a valid paged configuration, so it has to get past validation. Without
	// a checkpoint the next thing it does is fail on the model path, which names the file.
	err := cmdBench([]string{"--model", "unused.gguf", "--max-tokens", "1", "--blocks", "64",
		"--prefix-cache"})
	if err == nil || !strings.Contains(err.Error(), "unused.gguf") {
		t.Errorf("--blocks with --prefix-cache was rejected by validation: %v", err)
	}
}

// TestBenchPrefixCacheReportsReuseWithoutPanicking is the accepting half of the check above,
// on a real run rather than on the validation.
//
// The rejection test proves the two broken combinations stop early; this proves the same
// command still works when every flag is supplied, which is the half that a check written too
// broadly would break. It reaches the reuse note at the end of the run -- the last print, and
// the one that used to dereference a nil pool -- so a panic on the success path fails here.
func TestBenchPrefixCacheReportsReuseWithoutPanicking(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())
	// Long enough to fill more than one 16-position block, so the note's "nothing to share"
	// branch is not the one that happens to be exercised.
	prompt := strings.Repeat("word ", 40)

	out := captureStdout(t, func() {
		err := cmdBench([]string{"--model", path, "--max-tokens", "1", "--concurrency", "1",
			"--requests", "2", "--prefix-cache", "--blocks", "64", "--block-size", "16",
			"--prompt", prompt})
		if err != nil {
			t.Fatalf("cmdBench: %v", err)
		}
	})
	// The report must still describe a paged cache rather than falling back to the
	// contiguous line, which is how the ignored --block-size used to read.
	if !strings.Contains(out, "paged") {
		t.Errorf("output does not describe a paged cache:\n%s", out)
	}
	if !strings.Contains(out, "reused:") {
		t.Errorf("output has no reuse line:\n%s", out)
	}
}

// TestBenchSuiteRefusesFlagsItWouldIgnore covers the six settings --suite takes and then
// never reads.
//
// The suite fixes concurrency, request count, admission policy, and cache shape because those
// are the variables its rows compare: letting a flag override one would delete the comparison
// the table exists to make. Accepting the flag anyway was the defect. `--suite --policy bogus`
// ran the entire table and exited 0 without ever parsing the value, and `--suite --concurrency
// 32` printed eight scenarios measured at other concurrencies with nothing in the output to
// say the flag had been dropped -- so an operator had no way to tell a typo from a setting
// the suite honoured.
//
// The check walks fs.Visit rather than comparing values against defaults, and both cases below
// exist to prove it: --policy's default and --concurrency's default are valid values, and the
// test passes flags set to exactly those. A comparison against the default would accept both.
func TestBenchSuiteRefusesFlagsItWouldIgnore(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want []string
		why  string
	}{
		{
			name: "policy at its default value",
			args: []string{"--policy", "fifo"},
			want: []string{"--policy"},
			why:  "the suite picks a policy per scenario, so the value is never read",
		},
		{
			name: "concurrency at its default value",
			args: []string{"--concurrency", "1"},
			want: []string{"--concurrency"},
			why:  "the suite sets MaxActive per scenario, so the value is never read",
		},
		{
			name: "several at once, all named",
			args: []string{"--requests", "9", "--prefix-cache", "--block-size", "32"},
			want: []string{"--requests", "--prefix-cache", "--block-size"},
			why:  "an operator passing three wrong flags should learn about all three at once",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			args := append([]string{"--model", "unused.gguf", "--suite"}, c.args...)
			err := cmdBench(args)
			if err == nil {
				t.Fatalf("cmdBench accepted %v with --suite; %s", c.args, c.why)
			}
			// Every ignored flag has to be named: reporting the first one and stopping
			// turns a single fix into as many runs as there were flags.
			for _, w := range c.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error %q does not mention %s", err, w)
				}
			}
			if strings.Contains(err.Error(), "unused.gguf") {
				t.Errorf("error %q came from opening the model, so the flags were not checked", err)
			}
		})
	}

	// The flags the suite does read must stay accepted, or the check could pass by rejecting
	// every option. Without a checkpoint they fail on the model path, which names the file.
	err := cmdBench([]string{"--model", "unused.gguf", "--suite", "--max-tokens", "4",
		"--prompt", "Once upon a time", "--workers", "2"})
	if err == nil || !strings.Contains(err.Error(), "unused.gguf") {
		t.Errorf("--suite rejected the flags it does use: %v", err)
	}
}

// TestBenchSuiteRunsEndToEnd is the accepting half of the flag check above, on the path the
// check gates rather than on validation alone.
//
// Every existing suite test calls the scenarios directly, so nothing covered the command that
// wires them together: `cmdBench --suite` builds the prompt, sizes the table, and prints the
// conditions. The flag-validation change sits between the two, which is exactly where a
// check written slightly too broadly would trade a silent ignore for a broken command.
//
// The assertions are on structure, not on numbers: throughput on a fixture varies with the
// machine, so what has to hold is that every scenario produced a row, none of them failed,
// and the header names the conditions that make the numbers judgeable.
func TestBenchSuiteRunsEndToEnd(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())

	out := captureStdout(t, func() {
		err := cmdBench([]string{"--model", path, "--suite", "--max-tokens", "1",
			"--prompt", "Once upon a time", "--workers", "2"})
		if err != nil {
			t.Fatalf("cmdBench --suite: %v", err)
		}
	})

	// The conditions belong above the table: a throughput figure without them cannot be
	// compared against anything, which is what the suite exists to allow.
	for _, want := range []string{"model:", "prompt:", "generating:", "machine:", "go:", "SCENARIO"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing the %q line:\n%s", want, out)
		}
	}
	// A scenario that failed prints "failed:" in its NOTE column instead of numbers, so its
	// absence is what distinguishes a complete table from one that skipped a row.
	if strings.Contains(out, "failed:") {
		t.Errorf("a scenario failed:\n%s", out)
	}
	// Every row in the table, in the order scenarioList declares them.
	for _, name := range []string{
		"single stream", "batch x4", "batch x16", "paged x16",
		"prefix no reuse", "prefix reuse x4", "policy shortest", "policy fifo",
	} {
		if !strings.Contains(out, name) {
			t.Errorf("output missing the %q row:\n%s", name, out)
		}
	}
}

// TestRunRefusesFlagsAfterThePrompt covers the shape flag.Parse accepts silently.
//
// Go's flag package stops at the first non-flag, so `orbix run "Once upon a time" --stats`
// parsed no flags at all: it generated the default 128 tokens where 8 were asked for, with
// no stats line, and exited 0. Neither the ignored budget nor the missing output says which
// argument was the problem. `model inspect` already refuses this shape; run now does too,
// and the message names the flag that landed after the prompt.
func TestRunRefusesFlagsAfterThePrompt(t *testing.T) {
	err := cmdRun([]string{"--model", "unused.gguf", "Once upon a time", "--max-tokens", "8"})
	if err == nil {
		t.Fatal("run accepted a flag after the prompt, which flag.Parse silently drops")
	}
	for _, want := range []string{"--max-tokens", "after"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// TestRunPromptIsEveryRemainingWord covers the other half: a prompt typed without quotes.
//
// `orbix run Once upon a time` used to prefill only "Once" and discard the rest, which shows
// up as a prompt token count of 2 rather than 5 and is easy to mistake for tokenizer
// behaviour. The words are joined instead, and a single quoted word still yields itself.
func TestRunPromptIsEveryRemainingWord(t *testing.T) {
	cases := []struct {
		args []string
		want string
	}{
		{[]string{"Once", "upon", "a", "time"}, "Once upon a time"},
		{[]string{"Once upon a time"}, "Once upon a time"},
		{[]string{"Once"}, "Once"},
	}
	for _, c := range cases {
		fs := flag.NewFlagSet("run", flag.ContinueOnError)
		if err := fs.Parse(c.args); err != nil {
			t.Fatalf("Parse(%v): %v", c.args, err)
		}
		got, err := promptFromArgs(fs)
		if err != nil {
			t.Fatalf("promptFromArgs(%v): %v", c.args, err)
		}
		if got != c.want {
			t.Errorf("promptFromArgs(%v) = %q, want %q", c.args, got, c.want)
		}
	}

	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	if _, err := promptFromArgs(fs); err == nil {
		t.Error("promptFromArgs accepted no arguments")
	}
}

// TestFormatValueKeepsUTF8Whole covers the metadata table's truncation.
//
// formatValue cuts long values to a fixed byte width so one row stays readable, and the
// offset it used to cut at had no relation to rune boundaries. A metadata string is
// arbitrary UTF-8 -- a checkpoint name in Chinese, a chat template with an emoji -- so
// slicing at byte 57 can land inside a three-byte character and produce a string that is not
// valid UTF-8. Measured before the fix: "a" followed by twenty "中" is 61 bytes, and the
// result was 60 bytes that utf8.ValidString rejected. It reaches the terminal as mojibake and
// breaks a pipeline that parses the --json output.
//
// The sweep is over the byte lengths around the boundary rather than one case, because the
// previous cut was correct whenever the boundary happened to fall on a character edge: a
// single example could pass while the bug was present.
func TestFormatValueKeepsUTF8Whole(t *testing.T) {
	// Every prefix length from 0 to 2 and every count of three-byte characters that puts the
	// total within a few bytes of the 60-byte limit.
	for prefix := 0; prefix < 3; prefix++ {
		for n := 18; n <= 21; n++ {
			in := strings.Repeat("a", prefix) + strings.Repeat("中", n)
			got := formatValue(gguf.Value{Type: gguf.TypeString, Str: in})
			if !utf8.ValidString(got) {
				t.Errorf("formatValue(%d-byte %q) = %q, which is not valid UTF-8", len(in), in, got)
			}
			if len(got) > 60 {
				t.Errorf("formatValue(%d-byte input) = %d bytes, want at most 60", len(in), len(got))
			}
			// A value that fits is returned unchanged, multibyte or not.
			if len(in) <= 60 {
				if got != in {
					t.Errorf("formatValue(%q) = %q, want it unchanged", in, got)
				}
			} else if !strings.HasSuffix(got, "...") {
				t.Errorf("formatValue(%d-byte input) = %q, want an ellipsis suffix", len(in), got)
			}
		}
	}

	// The array form is a count, not a truncation, and stays as it was.
	got := formatValue(gguf.Value{Type: gguf.TypeArray, Array: make([]gguf.Value, 7)})
	if got != "7 values" {
		t.Errorf("formatValue(array) = %q, want \"7 values\"", got)
	}
}

// TestReportFormatsTheMessageAndTheStatus covers the mapping from an error to what a caller
// sees: the line on stderr and the exit status.
//
// The "orbix: " prefix is how a script and a human tell this program's complaint apart from
// whatever else was on the terminal -- it was in the message for every command and nothing
// asserted it, so removing it passed the cmd suite.
func TestReportFormatsTheMessageAndTheStatus(t *testing.T) {
	t.Run("success is silent", func(t *testing.T) {
		var w bytes.Buffer
		if got := report(nil, &w); got != 0 {
			t.Errorf("report(nil) = %d, want 0", got)
		}
		if w.Len() != 0 {
			t.Errorf("report(nil) wrote %q to stderr", w.String())
		}
	})

	t.Run("an error is prefixed and non-zero", func(t *testing.T) {
		var w bytes.Buffer
		err := errors.New("something went wrong")
		if got := report(err, &w); got != 1 {
			t.Errorf("report(error) = %d, want 1", got)
		}
		got := w.String()
		if !strings.HasPrefix(got, "orbix: ") {
			t.Errorf("stderr line %q does not start with the program name", got)
		}
		if !strings.Contains(got, "something went wrong") {
			t.Errorf("stderr line %q does not carry the message", got)
		}
	})

	t.Run("help is a success", func(t *testing.T) {
		// The flag package prints the usage itself and returns ErrHelp; treating that as a
		// failure made every command's -h exit 1 with an extra complaint on stderr.
		var w bytes.Buffer
		if got := report(flag.ErrHelp, &w); got != 0 {
			t.Errorf("report(flag.ErrHelp) = %d, want 0", got)
		}
		if w.Len() != 0 {
			t.Errorf("report(flag.ErrHelp) wrote %q, want nothing", w.String())
		}
	})

	t.Run("a wrapped help is still a success", func(t *testing.T) {
		// errors.Is is what makes this work through wrapping, which is how a command that
		// passes the flag package's error up reports it.
		var w bytes.Buffer
		if got := report(fmt.Errorf("serve: %w", flag.ErrHelp), &w); got != 0 {
			t.Errorf("report(wrapped ErrHelp) = %d, want 0", got)
		}
	})

	t.Run("a writer that fails does not change the status", func(t *testing.T) {
		// The write is best effort: a CLI reporting a failure has nothing left to report on
		// if stderr is gone, and the exit status is what a caller can act on.
		if got := report(errors.New("boom"), failingWriter{err: errors.New("closed")}); got != 1 {
			t.Errorf("report with a broken writer = %d, want 1", got)
		}
	})
}

// TestUsageAndDispatchAgree checks the help text against the dispatcher in both directions.
//
// A command that dispatches but is absent from the usage text is unreachable in practice: the
// only way to learn it exists is to read main.go. The reverse is a help line promising something
// the binary does not do. The existing usage test spot-checks three strings, so a newly added
// command that nobody documented -- or a documented one whose case was dropped -- still passed.
//
// Each name is exercised through run() rather than compared as text: dispatching is the property
// that matters, and a command that is listed but errors out from its own flag parsing has still
// been dispatched.
func TestUsageAndDispatchAgree(t *testing.T) {
	// The commands that appear under "Commands:" in the usage text.
	out := captureStdout(t, func() {
		if err := run(nil); err != nil {
			t.Fatalf("run(nil) = %v", err)
		}
	})
	var listed []string
	inCommands := false
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "Commands:") {
			inCommands = true
			continue
		}
		if inCommands {
			if strings.TrimSpace(line) == "" {
				break
			}
			if name, rest, ok := strings.Cut(strings.TrimSpace(line), " "); ok && rest != "" {
				listed = append(listed, name)
			}
		}
	}
	if len(listed) < 5 {
		t.Fatalf("parsed only %d commands from the usage text: %v", len(listed), listed)
	}

	for _, name := range listed {
		// "model inspect" is the one subcommand form; its name is "model" in the dispatcher.
		cmd := name
		if cmd == "model" {
			// Dispatched through its own subcommand check; a bare "model" is an error naming
			// the subcommand, which still proves the case exists.
			err := run([]string{"model"})
			if err == nil || !strings.Contains(err.Error(), "inspect") {
				t.Errorf("usage lists %q but run([model]) returned %v", name, err)
			}
			continue
		}
		// A command that reaches its own flag parsing is dispatched. -h makes that happen
		// without doing work, and report() maps the resulting ErrHelp to success.
		done := make(chan error, 1)
		go func() {
			defer func() { _ = recover() }()
			done <- run([]string{cmd, "-h"})
		}()
		select {
		case err := <-done:
			if err != nil && !errors.Is(err, flag.ErrHelp) {
				// A command may legitimately refuse -h or require a model first; what must not
				// happen is "unknown command", which is the dispatcher not having the case.
				if strings.Contains(err.Error(), "unknown command") {
					t.Errorf("usage lists %q but the dispatcher does not know it: %v", name, err)
				}
			}
		case <-time.After(20 * time.Second):
			// serve with -h exits immediately; a hang here would mean the command started
			// working, which this test cannot allow.
			t.Errorf("%q did not return from -h", name)
		}
	}

	// The reverse direction: every command the dispatcher knows must be listed. The set is
	// restated here rather than parsed out of main.go's switch -- the spelling is not stable
	// enough to read mechanically -- and the cost of that choice is that adding a command
	// means touching both main.go and this list. That is the point: the second edit is what
	// makes someone notice the usage text has not been updated, and both loops below fail
	// when the two sides drift.
	all := []string{"run", "bench", "chat", "serve", "model", "version", "help"}
	for _, name := range all {
		if !slices.Contains(listed, name) {
			t.Errorf("the dispatcher knows %q but the usage text does not list it", name)
		}
	}
	for _, name := range listed {
		if !slices.Contains(all, name) {
			t.Errorf("the usage text lists %q, which no test knows is dispatched", name)
		}
	}

	// And a name the dispatcher rejects must not be listed.
	if err := run([]string{"definitely-not-a-command"}); err == nil {
		t.Error("an unknown command was accepted")
	}
}

// TestVersionAcceptsBothFlagSpellings covers the alias the earlier version test missed.
//
// The defect this command had was that `version --json` printed the human report and exited
// zero -- a build script piping it into a JSON parser got a parse error where the flag said
// success. The test written for it called `version -json`, the single-dash spelling, so adding
// the other spelling to the case was never checked: narrowing it back to `-json` alone passes.
//
// Both spellings are used elsewhere in this CLI (every command's -h and --help are one case),
// which is what makes the single-dash-only version a defect rather than a choice.
func TestVersionAcceptsBothFlagSpellings(t *testing.T) {
	for _, flag := range []string{"-json", "--json"} {
		t.Run(flag, func(t *testing.T) {
			out := captureStdout(t, func() {
				if err := run([]string{"version", flag}); err != nil {
					t.Fatalf("version %s: %v", flag, err)
				}
			})
			var got map[string]any
			if err := json.Unmarshal([]byte(out), &got); err != nil {
				t.Fatalf("version %s did not produce JSON: %v\n%s", flag, err, out)
			}
			if _, ok := got["version"]; !ok {
				t.Errorf("version %s produced no version field: %v", flag, got)
			}
		})
	}

	for _, flag := range []string{"-h", "--help"} {
		t.Run(flag, func(t *testing.T) {
			out := captureStdout(t, func() {
				if err := run([]string{"version", flag}); err != nil {
					t.Fatalf("version %s: %v", flag, err)
				}
			})
			if !strings.Contains(out, "Usage:") {
				t.Errorf("version %s printed %q, want the usage line", flag, out)
			}
		})
	}
}

// TestVersionJSONOmitsAnUnknownCgoEnabled covers the omitempty on cgo_enabled.
//
// The field is a pointer so a binary with no build info to read it from can leave it out
// entirely rather than report a false. Dropping omitempty would print `"cgo_enabled": null`,
// which a script reading the field would have to special-case -- and `cgo` already answers the
// question for every binary, so null is worse than absent.
//
// The condition cannot be forced in-process, since a test binary always has build info. What is
// pinned instead is the encoding contract itself: a nil pointer produces no key, which is what
// the struct's tag asks for.
func TestVersionJSONOmitsAnUnknownCgoEnabled(t *testing.T) {
	type probe struct {
		CgoEnabled *string `json:"cgo_enabled,omitempty"`
	}
	b, err := json.Marshal(probe{})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if strings.Contains(string(b), "cgo_enabled") {
		t.Errorf("a nil cgo_enabled was encoded as %s, want the key absent", b)
	}

	// And when it is known, it is present as a string.
	v := "0"
	b, err = json.Marshal(probe{CgoEnabled: &v})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if !strings.Contains(string(b), `"cgo_enabled":"0"`) {
		t.Errorf("a known cgo_enabled encoded as %s", b)
	}
}

// captureStderr is captureStdout for the half of a report that goes to stderr.
func captureStderr(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stderr
	os.Stderr = w
	done := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		buf.ReadFrom(r)
		done <- buf.String()
	}()

	defer func() {
		os.Stderr = old
	}()
	fn()
	w.Close()
	os.Stderr = old
	return <-done
}

// TestRunStatsReport covers `run --stats`, which is the only place a caller sees the model's
// parameter count, its load time, and whether the window is what stopped generation.
//
// None of it had a test: reading the whole output as the completion, dropping the prompt
// prefix, or reporting a parameter count of zero all passed the cmd suite. The window note is
// the part a reader needs most -- a run that asked for 16 tokens and produced 9 otherwise looks
// like the model stopped early.
func TestRunStatsReport(t *testing.T) {
	// A 64-position window, so a prompt of a few tokens plus a large budget fills it and the
	// note has to appear.
	path := gguftest.Model{NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())

	var stdout string
	stderr := captureStderr(t, func() {
		stdout = captureStdout(t, func() {
			if err := run([]string{"run", "--model", path, "--stats", "--max-tokens", "200", "hi"}); err != nil {
				t.Fatalf("run: %v", err)
			}
		})
	})

	// The stats block goes to stderr, so stdout is the completion alone.
	for _, want := range []string{"model:", "params:", "load:", "prompt:", "gen:", "cache:"} {
		if !strings.Contains(stderr, want) {
			t.Errorf("stats report is missing %q:\n%s", want, stderr)
		}
	}
	// A parameter count of zero would be a fabricated number; the fixture has weights.
	if strings.Contains(stderr, "params: 0\n") {
		t.Errorf("stats report claims zero parameters:\n%s", stderr)
	}

	// The completion on stdout must not repeat the prompt: the prompt is echoed only with
	// --echo, which is not set here.
	if strings.Contains(stdout, "hi") && strings.Count(stdout, "hi") > 0 {
		// The fixture's output may coincidentally contain "hi", so this is checked through the
		// prompt's own words instead: the whole prompt would appear as a prefix.
		if strings.HasPrefix(stdout, "hi") {
			t.Errorf("stdout starts with the prompt, so the completion slice is wrong: %q", stdout)
		}
	}

	// The window note appears when the budget was cut by the window rather than by EOS.
	if !strings.Contains(stderr, "window is full") {
		t.Errorf("a run that filled the window did not say so:\n%s", stderr)
	}
}

// TestBenchRefusesAPromptPastTheContext covers both bench paths that encode the prompt.
//
// The single-measurement path and --suite each encode the prompt and refuse one that fills the
// window. Neither had a test: the refusal is a plain length comparison, but it is the difference
// between an error naming both numbers and an engine failure from inside the prefill, and the two
// paths duplicate it, so a fix to one would not reach the other.
//
// The prompt is long enough to encode past the fixture's 128-position window whatever the
// tokenizer does with it.
func TestBenchRefusesAPromptPastTheContext(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 128}.Write(t, t.TempDir())
	// Exactly 128 tokens once encoded -- measured, not derived -- so the >= boundary is what the
	// test exercises. A prompt far past the window cannot distinguish >= from >.
	long := strings.Repeat("a ", 123) + "done"

	for _, c := range []struct {
		name string
		args []string
	}{
		{"single", []string{"bench", "--model", path, "--prompt", long}},
		{"suite", []string{"bench", "--model", path, "--suite", "--prompt", long}},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := run(c.args)
			if err == nil {
				t.Fatal("bench accepted a prompt past the context")
			}
			// Both numbers belong in the message: the prompt's length alone does not say the
			// window is smaller.
			if !strings.Contains(err.Error(), "context is 128") {
				t.Errorf("error %q does not name the context", err)
			}
			if !strings.Contains(err.Error(), "prompt is") {
				t.Errorf("error %q does not name the prompt's length", err)
			}
		})
	}
}

// TestBenchReportsWhetherPrefixReuseHappened covers the two lines bench prints about reuse.
//
// --prefix-cache is the one flag whose effect is invisible in a throughput number: a run where
// sharing worked and one where it could not apply print the same tok/s. The two messages exist to
// tell them apart, and neither had a test -- a run that reused nothing printed nothing, which is
// how the flag looked unimplemented.
//
// The sequential case is what makes reuse possible: runRequests submits every request at once, so
// with MaxActive as large as the count they all prefill together and none can attach a prefix
// another has not written yet.
func TestBenchReportsWhetherPrefixReuseHappened(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	// Spans several 16-position blocks, so there is a whole block to publish and reuse.
	prompt := strings.Repeat("Once upon a time there was a girl. ", 4)

	t.Run("reuse that happened is counted", func(t *testing.T) {
		out := captureStdout(t, func() {
			err := run([]string{"bench", "--model", path, "--prompt", prompt,
				"--prefix-cache", "--blocks", "64", "--block-size", "16",
				"--requests", "2", "--concurrency", "1", "--max-tokens", "2"})
			if err != nil {
				t.Fatalf("bench: %v", err)
			}
		})
		if !strings.Contains(out, "prompt positions") {
			t.Errorf("bench did not report reused positions:\n%s", out)
		}
		if strings.Contains(out, "none") {
			t.Errorf("bench reported no reuse for a sequential run with a spanning prompt:\n%s", out)
		}
	})

	t.Run("reuse that could not apply says why", func(t *testing.T) {
		// A prompt shorter than one block leaves nothing to publish, so reuse cannot happen
		// however many requests run. This is the case the message is about -- and it is NOT the
		// concurrency: measured, four concurrent requests over a prompt spanning several blocks
		// reported 240 reused positions, because the batched prefill still publishes blocks for
		// the requests processed first in the batch.
		out := captureStdout(t, func() {
			err := run([]string{"bench", "--model", path, "--prompt", "hi",
				"--prefix-cache", "--blocks", "64", "--block-size", "16",
				"--requests", "4", "--concurrency", "4", "--max-tokens", "2"})
			if err != nil {
				t.Fatalf("bench: %v", err)
			}
		})
		if !strings.Contains(out, "reused:") {
			t.Errorf("bench printed no reuse line:\n%s", out)
		}
		if !strings.Contains(out, "none") {
			t.Errorf("bench claimed reuse for a prompt shorter than one block:\n%s", out)
		}
		// The explanation is the actionable part: it names the block size so the reader can
		// lengthen the prompt.
		if !strings.Contains(out, "block") {
			t.Errorf("the message does not mention the block:\n%s", out)
		}
	})
}

// TestBenchSuiteReportsWhichScenariosFailed covers the aggregation at the end of a suite.
//
// A scenario failing prints "failed:" in its NOTE column, and the whole run must also return an
// error. Without that the exit status is zero for a table that visibly contains a failure, which a
// script checking $? would read as a successful benchmark -- the numbers it captured would be the
// ones that did run, silently missing a row.
//
// The suite always uses scenarioList, so this drives runScenarios with an explicit pair: one entry
// that returns a result and one that fails. The names in the message are what a reader uses to find
// the row.
func TestBenchSuiteReportsWhichScenariosFailed(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())
	opts := suiteOptions{model: path, prompt: []int32{1, 2, 3}, maxTok: 1, workers: 1}

	scenarios := []scenario{
		{name: "working", run: func(suiteOptions) (result, error) {
			return result{tokens: 1, elapsed: time.Millisecond}, nil
		}},
		{name: "broken", run: func(suiteOptions) (result, error) {
			return result{}, errors.New("simulated failure")
		}},
	}

	var err error
	out := captureStdout(t, func() { err = runScenarios(opts, scenarios) })
	if err == nil {
		t.Fatalf("a suite with a failing scenario returned nil:\n%s", out)
	}
	// The message has to say how many failed and name them, which is what locates the row.
	if !strings.Contains(err.Error(), "1 scenario(s) failed") {
		t.Errorf("error %q does not count the failures", err)
	}
	if !strings.Contains(err.Error(), "broken") {
		t.Errorf("error %q does not name the failed scenario", err)
	}
	if strings.Contains(err.Error(), "working") {
		t.Errorf("error %q names a scenario that succeeded", err)
	}
	// Both rows still appear in the table, so the successful one's numbers are not lost.
	for _, want := range []string{"working", "broken", "failed: simulated failure"} {
		if !strings.Contains(out, want) {
			t.Errorf("table missing %q:\n%s", want, out)
		}
	}

	// The all-good case returns nil, so the error means something.
	scenarios = scenarios[:1]
	out = captureStdout(t, func() { err = runScenarios(opts, scenarios) })
	if err != nil {
		t.Errorf("a suite with no failures returned %v:\n%s", err, out)
	}
}
