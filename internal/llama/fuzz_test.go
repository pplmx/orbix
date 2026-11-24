package llama

import (
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
)

// FuzzLoadConfig checks that no metadata makes the loader panic or accept an impossible model.
//
// This package is the first that reads a checkpoint's metadata for meaning rather than for
// structure: LoadConfig looks up two dozen keys, divides by them and compares the results against
// each other. gguf.Read is fuzzed separately, so a truncated container is already covered; what is
// not is a container that reads cleanly and still describes an impossible model.
//
// The inputs are the numbers rather than the file, which is the correction to a first version of
// this fuzz that fed whole files through gguf.Read first. Random bytes essentially never survive
// that parse, so every execution returned before reaching LoadConfig -- the fuzzer was measuring
// the container parser a second time. Fuzzing the hyperparameters directly is the abstraction
// level at which the arithmetic under test lives.
func FuzzLoadConfig(f *testing.F) {
	// The smallest shape the loader accepts, and a few that are known to be rejected. The seeds
	// matter less here than usual: the interesting inputs are numbers, and the corpus generator
	// reaches them quickly.
	f.Add("llama", uint32(32), uint32(2), uint32(2), uint32(2))
	f.Add("qwen3", uint32(64), uint32(4), uint32(4), uint32(2))
	f.Add("gemma3", uint32(48), uint32(1), uint32(3), uint32(1))
	// Zero and mismatched values: refused, and the point is that they are refused rather than
	// panicking on a division.
	f.Add("llama", uint32(0), uint32(1), uint32(1), uint32(1))
	f.Add("llama", uint32(32), uint32(1), uint32(0), uint32(1))
	f.Add("llama", uint32(32), uint32(1), uint32(7), uint32(3))

	f.Fuzz(func(t *testing.T, arch string, hidden, layers, heads, kvHeads uint32) {
		md := map[string]gguf.Value{
			"general.architecture":            {Type: gguf.TypeString, Str: arch},
			arch + ".embedding_length":        u32Val(hidden),
			arch + ".block_count":             u32Val(layers),
			arch + ".attention.head_count":    u32Val(heads),
			arch + ".attention.head_count_kv": u32Val(kvHeads),
			arch + ".feed_forward_length":     u32Val(hidden),
			arch + ".context_length":          u32Val(layers),
		}
		_, err := LoadConfig(&gguf.File{Metadata: md})
		if err == nil {
			// Unreachable from these inputs, and said so rather than asserted around: LoadConfig
			// also requires a tokenizer vocabulary and a tensor directory, and this fuzz builds
			// only the seven numeric keys. A config accepted from them is therefore a change in
			// what the loader requires, which is worth failing on rather than tolerating.
			t.Fatal("LoadConfig accepted a file with no vocabulary and no tensors")
		}
		// Every rejection has to name something. An empty message would leave a user with no
		// idea which key is wrong, and the checks in this function are the only place a
		// malformed checkpoint is explained.
		if err.Error() == "" {
			t.Fatal("LoadConfig returned an empty error message")
		}
	})
}
