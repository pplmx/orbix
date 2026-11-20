package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"
	"text/tabwriter"
	"unicode/utf8"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/tokenize"
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
	fmt.Fprintf(tw, "Metadata entries\t%d\n", f.MetadataCount())
	fmt.Fprintf(tw, "Tensors\t%d\n", f.TensorCount())
	fmt.Fprintf(tw, "Tensor data offset\t%d\n", f.TensorDataLoc)

	if archErr == nil {
		printHyperparams(tw, f, arch)
	}
	printTokenizerLine(tw, f)
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

// printTokenizerLine prints the pre-tokenizer the file names and whether this build
// reproduces it.
//
// The second half is the part worth printing. A family that is not implemented tokenizes as
// one chunk, which produces different tokens from the ones the model was trained on and
// shows up as degraded output rather than as an error -- so a reader needs to see it before
// judging the model's quality, not after.
func printTokenizerLine(tw *tabwriter.Writer, f *gguf.File) {
	tok, err := tokenize.New(f)
	if err != nil {
		fmt.Fprintf(tw, "\nTokenizer\t(unusable: %v)\n", err)
		return
	}
	model, _ := f.String("tokenizer.ggml.model")
	name := tok.PreName()
	if name == "" {
		name = "(none)"
	}
	fmt.Fprintf(tw, "\nTokenizer\t%s, pre-tokenizer %s\n", model, name)

	switch {
	case tok.PreImplemented():
		fmt.Fprintf(tw, "Pre-tokenizer\timplemented\n")
	case tok.PreKnown():
		fmt.Fprintf(tw, "Pre-tokenizer\tNOT implemented -- text is split as one chunk, "+
			"which differs from what this model was trained on\n")
	default:
		fmt.Fprintf(tw, "Pre-tokenizer\tNOT a name this package knows; check "+
			"tokenizer.ggml.pre (splitting is disabled)\n")
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
	for ti := range f.Tensors() {
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
	slices.SortFunc(order, func(a, b gguf.QuantType) int {
		return strings.Compare(a.String(), b.String())
	})

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
	if f.TensorCount() == 0 {
		return nil
	}
	fmt.Fprintln(tw)
	fmt.Fprintf(tw, "Tensors (%d)\n", f.TensorCount())
	fmt.Fprintln(tw, "  Name\tType\tShape\tSize")
	for ti := range f.Tensors() {
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
//
// Metadata values keep their GGUF type: an integer is a JSON number, a bool is a JSON bool, a
// string is a JSON string. Rendering every value with Value.String made the output a map of
// strings, so a script reading llama.context_length had to parse "128" back out of it -- which
// is the work a machine-readable flag exists to remove. The scalar projection below is the one
// place the mapping is decided.
//
// Arrays stay a summary -- "[32000 string]" -- rather than being expanded. A tokenizer carries
// three arrays of 32000 entries, and inlining them would turn a 200-line document into
// hundreds of thousands with no one reading them; the type and length are what a script needs
// to know the array is there, and a script that needs the contents reads the file with the
// loader.
func writeJSON(w io.Writer, f *gguf.File) error {
	type tensor struct {
		Name   string   `json:"name"`
		Dims   []uint64 `json:"dims"`
		Type   string   `json:"type"`
		Offset uint64   `json:"offset"`
		Bytes  *int64   `json:"bytes,omitempty"`
	}
	out := struct {
		Version       uint32         `json:"version"`
		Alignment     int64          `json:"alignment"`
		TensorDataLoc int64          `json:"tensor_data_offset"`
		Metadata      map[string]any `json:"metadata"`
		Tensors       []tensor       `json:"tensors"`
	}{
		Version:       f.Version,
		Alignment:     f.Alignment,
		TensorDataLoc: f.TensorDataLoc,
		Metadata:      make(map[string]any, f.MetadataCount()),
		Tensors:       make([]tensor, 0, f.TensorCount()),
	}
	for k, v := range f.Metadata {
		out.Metadata[k] = jsonValue(v)
	}
	for ti := range f.Tensors() {
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

// jsonValue projects one GGUF metadata value onto the JSON type that matches it.
//
// The switch is on the GGUF type rather than on the populated field, because the type is what
// the file records: a uint64 that happens to hold a small number is still an unsigned integer,
// and a caller comparing against a Go int should see the same width the file declares.
func jsonValue(v gguf.Value) any {
	switch v.Type {
	case gguf.TypeString:
		return v.Str
	case gguf.TypeBool:
		return v.Bool
	case gguf.TypeUint8, gguf.TypeUint16, gguf.TypeUint32, gguf.TypeUint64:
		return v.Uint
	case gguf.TypeInt8, gguf.TypeInt16, gguf.TypeInt32, gguf.TypeInt64:
		return v.Int
	case gguf.TypeFloat32, gguf.TypeFloat64:
		// A non-finite float has no JSON representation: encoding/json refuses NaN and both
		// infinities, and refusing them is correct -- JSON's number grammar has no token for
		// any of the three. The file is still legal, though: a GGUF float is any 64-bit
		// pattern, and the reader accepts whatever the bits say, so a checkpoint carrying a
		// NaN (a broken conversion, a sentinel for "unset") can exist and this command has to
		// describe it rather than exit with an error that names no key. Measured: a file with
		// test.nan made `model inspect --json` print "orbix: json: unsupported value: NaN" and
		// nothing else.
		//
		// Emitted as the string strconv produces ("NaN", "+Inf", "-Inf"), which is the one
		// spelling that survives a JSON round trip while still reading as the value it is. A
		// null would decode as "absent" and a 0 as a real number, and both lose the fact.
		if math.IsNaN(v.Float) || math.IsInf(v.Float, 0) {
			return strconv.FormatFloat(v.Float, 'g', -1, 64)
		}
		// The float's own precision, as a number rather than the string FormatFloat builds.
		// A float32 cannot represent more than its type's digits, and encoding/json would
		// otherwise print the widened float64 in full -- 9.999999747378752e-06 for 1e-5.
		f, err := strconv.ParseFloat(gguf.FormatFloat(v.Type, v.Float), 64)
		if err != nil {
			return v.Float
		}
		return f
	default:
		// Arrays and anything unknown keep the summary form, which is what the text output
		// shows and what a script needs to know an array is there.
		return v.String()
	}
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
//
// The cut is measured in bytes but must land on a rune boundary: a metadata string is
// arbitrary UTF-8 -- a checkpoint name in Chinese, a template with an emoji -- and slicing at
// a fixed byte offset splits a multi-byte character, which writes invalid UTF-8 to the
// terminal and makes the --json output fail to parse downstream. Measured before this: a
// 61-byte name whose first character is ASCII produced a 60-byte result that
// utf8.ValidString rejected. The truncation budget includes the ellipsis, so the result
// stays inside the same width the fixed cut had.
func formatValue(v gguf.Value) string {
	if v.Type == gguf.TypeArray {
		return fmt.Sprintf("%d values", len(v.Array))
	}
	const max = 60
	s := v.String()
	if len(s) <= max {
		return s
	}
	// Keep max-3 bytes plus the ellipsis, retreating to the start of the last rune that
	// would otherwise be split. utf8.RuneStart walks back over continuation bytes, whose
	// top two bits are 10.
	keep := max - 3
	for keep > 0 && !utf8.RuneStart(s[keep]) {
		keep--
	}
	return s[:keep] + "..."
}
