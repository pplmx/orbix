package quant

import (
	"math"
	"os"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
)

// TestTensorDecodesTheSameTwice pins the properties a lazy decoder depends on.
//
// Today every tensor is decoded once, up front, and the model keeps the float32 values. Decoding
// per matmul instead -- the optimization LoadWeights names as the one to make later -- reads the
// same tensor many times over a run, and it is only a safe refactor if decoding is a pure function
// of the file's bytes returning a slice the caller owns. Three properties follow, and each is
// checked because each fails on its own:
//
//  1. the second call returns the same values as the first, bit for bit;
//  2. writing through the returned slice does not change what the next call returns;
//  3. two tensors' results do not share a buffer, so writing one does not reach the other.
//
// Bit equality rather than a tolerance: the same bytes through the same arithmetic have no reason
// to differ, and a tolerance would permit exactly the drift a stateful decoder produces.
//
// The second property is the one the doc comment on Tensor states and nothing tested. A decoder
// that kept a scratch buffer and handed it out would satisfy the first and fail this, and that is
// not hypothetical in this repository -- reusing a scratch slice across calls is what broke
// PagedKVCache.spans() on its parallel path.
//
// It used to be gated on MODEL, on the grounds that it needed a checkpoint's real mix of types.
// The fixture has that mix: four types whose decoders share no code path (a word copy, two
// 32-weight block formats, and one 256-weight super-block format). MODEL is still read when set,
// because a checkpoint has hundreds of tensors and the fixture four, but a machine without one now
// checks the property instead of skipping it.
func TestTensorDecodesTheSameTwice(t *testing.T) {
	t.Run("fixture", func(t *testing.T) {
		path := mixedTypeFixture(t)
		checkDecodingTwice(t, path)
		checkTensorsDoNotShareABuffer(t, path)
	})

	// A real file is a superset of the fixture's shapes rather than a different property, so its
	// absence is not a reason to skip anything. The cross-tensor check stays on the fixture: it
	// reads every element of every tensor four times, which for a checkpoint is tens of gigabytes
	// of decodes for a property that per-tensor allocation already makes obvious.
	if path := os.Getenv("MODEL"); path != "" {
		t.Run("MODEL", func(t *testing.T) { checkDecodingTwice(t, path) })
	}
}

// mixedTypeFixture writes one tensor of each of four types, laid out in one data section.
//
// The four are chosen for having separate decoders: F32 is a word copy, Q4_0 and Q8_0 are
// 32-weight blocks with different bit packing, and Q4_K is a 256-weight super-block whose scales
// are themselves packed across three arrays. State carried between calls would have to live in one
// of those four decoders, and they have nowhere to share it.
func mixedTypeFixture(t *testing.T) string {
	t.Helper()
	var data []byte

	f32Off := uint64(len(data))
	for i := range 8 {
		data = appendF32(data, float32(i+1))
	}
	q4Off := uint64(len(data))
	data = append(data, q4_0Block(t, fill(32, 9), 1)...)
	q8Off := uint64(len(data))
	data = append(data, appendF16Bytes(nil, 0.25)...)
	for j := range 32 {
		data = append(data, byte(int8(j-16)))
	}
	q4kOff := uint64(len(data))
	src := make([]float32, 256)
	for i := range src {
		src[i] = float32(math.Sin(float64(i))) * 0.5
	}
	data = append(data, encodeQ4K(src)...)

	return writeModel(t, []gguf.TensorInfo{
		{Name: "f32", Dims: []uint64{8}, Type: gguf.F32, Offset: f32Off},
		{Name: "q4_0", Dims: []uint64{32}, Type: gguf.Q4_0, Offset: q4Off},
		{Name: "q8_0", Dims: []uint64{32}, Type: gguf.Q8_0, Offset: q8Off},
		{Name: "q4_k", Dims: []uint64{256}, Type: gguf.Q4_K, Offset: q4kOff},
	}, data)
}

func appendF32(dst []byte, v float32) []byte {
	bits := math.Float32bits(v)
	return append(dst, byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24))
}

// checkDecodingTwice reads every tensor of path three times: the first two must agree bit for bit,
// and the third must be unchanged after the caller writes through the first result.
func checkDecodingTwice(t *testing.T, path string) {
	t.Helper()
	l, err := OpenLoader(path)
	if err != nil {
		t.Fatalf("OpenLoader(%s): %v", path, err)
	}
	defer closeLoader(t, l)

	for _, name := range tensorNames(t, l) {
		first, err := l.Tensor(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		second, err := l.Tensor(name)
		if err != nil {
			t.Fatalf("%s on the second call: %v", name, err)
		}
		if len(first) != len(second) {
			t.Errorf("%s: %d values then %d", name, len(first), len(second))
			continue
		}
		for i := range first {
			if math.Float32bits(first[i]) != math.Float32bits(second[i]) {
				t.Errorf("%s: value %d is %v on one call and %v on the next",
					name, i, first[i], second[i])
				break
			}
		}

		if len(second) == 0 {
			continue
		}
		// Snapshot by value before the write. A decoder that caches its result hands the same
		// slice back both times, so writing through first is writing through second as well --
		// which is exactly why the comparison below has to be against a copy taken first.
		want0 := math.Float32bits(second[0])
		first[0] = math.Float32frombits(0x7F800000) // +Inf: no weight in either file is infinite
		third, err := l.Tensor(name)
		if err != nil {
			t.Fatalf("%s on the third call: %v", name, err)
		}
		if len(third) != len(second) {
			t.Errorf("%s: %d values after a write, %d before", name, len(third), len(second))
			continue
		}
		if math.Float32bits(third[0]) != want0 {
			t.Errorf("%s: writing to the returned slice changed the next call's value 0 from %v "+
				"to %v; the loads share a buffer", name,
				math.Float32frombits(want0), third[0])
		}
	}
}

// checkTensorsDoNotShareABuffer writes a sentinel into each tensor's result in turn and confirms no
// other tensor's next read exposes it.
//
// The per-tensor check above cannot see this: a decoder reusing one buffer for every tensor would
// pass it as long as it re-decoded on each call. It runs on the fixture only, because the check
// reads every element of every tensor for each write.
func checkTensorsDoNotShareABuffer(t *testing.T, path string) {
	t.Helper()
	l, err := OpenLoader(path)
	if err != nil {
		t.Fatalf("OpenLoader(%s): %v", path, err)
	}
	defer closeLoader(t, l)

	const sentinel = 0x7F800000 // +Inf
	names := tensorNames(t, l)
	got := make(map[string][]float32, len(names))
	orig := make(map[string]uint32, len(names))
	for _, name := range names {
		v, err := l.Tensor(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		got[name] = v
		if len(v) > 0 {
			orig[name] = math.Float32bits(v[0])
		}
	}

	for _, written := range names {
		if len(got[written]) == 0 {
			continue
		}
		got[written][0] = math.Float32frombits(sentinel)
		for _, other := range names {
			if other == written {
				continue
			}
			after, err := l.Tensor(other)
			if err != nil {
				t.Fatalf("%s: %v", other, err)
			}
			for i := range after {
				if math.Float32bits(after[i]) == sentinel {
					t.Errorf("%s[%d] holds the value written into %s; the two results share a "+
						"buffer", other, i, written)
					break
				}
			}
		}
		// Put the value back, or the next iteration reports the sentinel this one wrote as
		// evidence that the next tensor shares a buffer with this one.
		got[written][0] = math.Float32frombits(orig[written])
	}
}

func tensorNames(t *testing.T, l *Loader) []string {
	t.Helper()
	var names []string
	for ti := range l.File().Tensors() {
		names = append(names, ti.Name)
	}
	if len(names) == 0 {
		t.Fatal("the file carries no tensors, so this checked nothing")
	}
	return names
}
