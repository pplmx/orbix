package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
		Version       uint32            `json:"version"`
		Alignment     int64             `json:"alignment"`
		TensorDataLoc int64             `json:"tensor_data_offset"`
		Metadata      map[string]string `json:"metadata"`
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
	if _, ok := got.Metadata["general.architecture"]; !ok {
		t.Error("metadata missing general.architecture")
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
