package main

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/pplmx/orbix/internal/gguf"
)

// inspectOptions controls what `orbix model inspect` prints.
type inspectOptions struct {
	// JSON switches to machine-readable output.
	JSON bool
}

// inspect writes a summary of a GGUF file to w.
//
// Writing through an io.Writer rather than straight to os.Stdout means a
// short write or a closed pipe is reported as an error instead of being
// silently dropped, and it lets the tests capture output without touching the
// process's real stdout.
func inspect(w io.Writer, f *gguf.File, path string, opts inspectOptions) error {
	if opts.JSON {
		return writeJSON(w, f)
	}
	return writeHuman(w, f, path)
}

// writeHuman renders the human-readable summary.
func writeHuman(w io.Writer, f *gguf.File, path string) error {
	// tabwriter buffers until Flush, so an error can surface on either the
	// Fprintf calls or the Flush. Accumulate into a checked writer and report
	// the first failure once, instead of ignoring each one.
	ew := &errWriter{w: w}
	tw := tabwriter.NewWriter(ew, 0, 0, 2, ' ', 0)

	name, _ := f.String("general.name")
	if name == "" {
		name = gguf.BaseName(path)
	}
	arch, archErr := f.Arch()

	fmt.Fprintf(tw, "File\t%s\n", path)
	fmt.Fprintf(tw, "Name\t%s\n", name)
	if archErr == nil {
		fmt.Fprintf(tw, "Architecture\t%s\n", arch)
	} else {
		fmt.Fprintf(tw, "Architecture\t(unknown)\n")
	}
	fmt.Fprintf(tw, "GGUF version\t%d\n", f.Version)
	fmt.Fprintf(tw, "Alignment\t%d bytes\n", f.Alignment)
	fmt.Fprintf(tw, "Metadata entries\t%d\n", len(f.Metadata))
	fmt.Fprintf(tw, "Tensors\t%d\n", len(f.Tensors))
	fmt.Fprintf(tw, "Tensor data offset\t%d\n", f.TensorDataLoc)

	if archErr == nil {
		printHyperparams(tw, f, arch)
	}
	tw.Flush()
	if err := ew.err; err != nil {
		return err
	}

	if err := printQuantization(tw, ew, f); err != nil {
		return err
	}
	return printTensors(tw, ew, f)
}

// printHyperparams prints the architecture-specific keys a reader needs to
// make sense of the tensor shapes.
func printHyperparams(tw *tabwriter.Writer, f *gguf.File, arch string) {
	keys := []struct {
		label string
		key   string
	}{
		{"Layers", arch + ".block_count"},
		{"Embedding dim", arch + ".embedding_length"},
		{"Feed-forward dim", arch + ".feed_forward_length"},
		{"Attention heads", arch + ".attention.head_count"},
		{"KV heads", arch + ".attention.head_count_kv"},
		{"Context length", arch + ".context_length"},
		{"RoPE dim", arch + ".rope.dimension_count"},
		{"RoPE base", arch + ".rope.freq_base"},
		{"RMS norm epsilon", arch + ".attention.layer_norm_rms_epsilon"},
	}
	printed := false
	for _, k := range keys {
		v, ok := f.Lookup(k.key)
		if !ok {
			continue
		}
		if !printed {
			fmt.Fprintln(tw)
			fmt.Fprintln(tw, "Hyperparameters")
			printed = true
		}
		fmt.Fprintf(tw, "  %s\t%s\n", k.label, formatValue(v))
	}
}

// printQuantization summarizes how many tensors use each data type and how
// much space each type takes, which is the quickest answer to "how is this
// model quantized?".
func printQuantization(tw *tabwriter.Writer, ew *errWriter, f *gguf.File) error {
	type stat struct {
		count  int
		bytes  int64
		unsure bool
	}
	stats := map[gguf.QuantType]*stat{}
	var order []gguf.QuantType
	for _, ti := range f.Tensors {
		s, ok := stats[ti.Type]
		if !ok {
			s = &stat{}
			stats[ti.Type] = s
			order = append(order, ti.Type)
		}
		s.count++
		if n, ok := ti.Bytes(); ok {
			s.bytes += n
		} else {
			s.unsure = true
		}
	}
	if len(order) == 0 {
		return nil
	}
	// Sort by name so the table is stable across runs; map iteration order
	// would otherwise reshuffle rows.
	sort.Slice(order, func(i, j int) bool { return order[i].String() < order[j].String() })

	fmt.Fprintln(tw)
	fmt.Fprintln(tw, "Quantization")
	fmt.Fprintln(tw, "  Type\tTensors\tSize")
	for _, q := range order {
		s := stats[q]
		size := formatBytes(s.bytes)
		if s.unsure {
			size += " (partial)"
		}
		fmt.Fprintf(tw, "  %s\t%d\t%s\n", q, s.count, size)
	}
	tw.Flush()
	return ew.err
}

// printTensors lists each tensor with its shape, type, and encoded size.
func printTensors(tw *tabwriter.Writer, ew *errWriter, f *gguf.File) error {
	if len(f.Tensors) == 0 {
		return nil
	}
	fmt.Fprintln(tw)
	fmt.Fprintf(tw, "Tensors (%d)\n", len(f.Tensors))
	fmt.Fprintln(tw, "  Name\tType\tShape\tSize")
	for _, ti := range f.Tensors {
		shape := formatShape(ti.Dims)
		size := "?"
		if n, ok := ti.Bytes(); ok {
			size = formatBytes(n)
		}
		fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\n", ti.Name, ti.Type, shape, size)
	}
	tw.Flush()
	return ew.err
}

// writeJSON prints the metadata and tensor directory as JSON, for scripts.
func writeJSON(w io.Writer, f *gguf.File) error {
	type tensor struct {
		Name   string   `json:"name"`
		Dims   []uint64 `json:"dims"`
		Type   string   `json:"type"`
		Offset uint64   `json:"offset"`
		Bytes  *int64   `json:"bytes,omitempty"`
	}
	out := struct {
		Version       uint32            `json:"version"`
		Alignment     int64             `json:"alignment"`
		TensorDataLoc int64             `json:"tensor_data_offset"`
		Metadata      map[string]string `json:"metadata"`
		Tensors       []tensor          `json:"tensors"`
	}{
		Version:       f.Version,
		Alignment:     f.Alignment,
		TensorDataLoc: f.TensorDataLoc,
		Metadata:      map[string]string{},
		Tensors:       make([]tensor, 0, len(f.Tensors)),
	}
	for k, v := range f.Metadata {
		out.Metadata[k] = v.String()
	}
	for _, ti := range f.Tensors {
		t := tensor{Name: ti.Name, Dims: ti.Dims, Type: ti.Type.String(), Offset: ti.Offset}
		if n, ok := ti.Bytes(); ok {
			t.Bytes = &n
		}
		out.Tensors = append(out.Tensors, t)
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

// errWriter remembers the first write error, so a sequence of prints can be
// checked once instead of at every call site.
type errWriter struct {
	w   io.Writer
	err error
}

func (e *errWriter) Write(p []byte) (int, error) {
	if e.err != nil {
		return 0, e.err
	}
	n, err := e.w.Write(p)
	if err != nil {
		e.err = err
	}
	return n, err
}

// formatShape renders dims in the conventional outermost-first order, which
// is how model shapes are usually written even though ggml stores them
// innermost-first.
func formatShape(dims []uint64) string {
	if len(dims) == 0 {
		return "scalar"
	}
	parts := make([]string, len(dims))
	for i, d := range dims {
		parts[len(dims)-1-i] = fmt.Sprint(d)
	}
	return strings.Join(parts, " x ")
}

// formatBytes renders a byte count with binary units.
func formatBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.2f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

// formatValue renders a metadata value compactly, collapsing long arrays to a
// count so one table stays readable.
func formatValue(v gguf.Value) string {
	if v.Type == gguf.TypeArray {
		return fmt.Sprintf("%d values", len(v.Array))
	}
	const max = 60
	if s := v.String(); len(s) > max {
		return s[:max-3] + "..."
	} else {
		return s
	}
}
