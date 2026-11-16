package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
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
