package llama

import (
	"fmt"
	"maps"
	"math"
	"slices"

	"github.com/pplmx/orbix/internal/quant"
)

// layer is the weight set of one transformer block.
//
// Weights are stored dequantized, in ggml's row-major layout: a projection is a
// slice of out rows, each holding in contiguous weights. That is the orientation
// the kernels expect, so the forward pass never transposes anything.
type layer struct {
	attnNorm []float32 // [hidden]
	wq       []float32 // [hidden, heads*headDim]
	wk       []float32 // [hidden, kvHeads*headDim]
	wv       []float32 // [hidden, kvHeads*headDim]
	wo       []float32 // [heads*headDim, hidden]

	// qBias, kBias, and vBias are added to the projections before attention. They
	// are nil for architectures that do not use them (Llama) and present for those
	// that do (Qwen2). Keeping them separate from the weights rather than folding
	// them into the matrices is what makes a nil check the only branch the forward
	// pass needs.
	qBias []float32 // [heads*headDim]
	kBias []float32 // [kvHeads*headDim]
	vBias []float32 // [kvHeads*headDim]

	ffnNorm []float32 // [hidden]
	wGate   []float32 // [hidden, ffn]
	wUp     []float32 // [hidden, ffn]
	wDown   []float32 // [ffn, hidden]

	// MoE fields. Those experts are nil for a dense layer, which is what the forward
	// pass branches on: the presence of a router is the difference between running one
	// FFN and selecting several.
	//
	// The expert tensors are stored per expert rather than as one block, so the forward
	// pass hands matmul a contiguous matrix instead of a slice with an expert stride to
	// explain. Splitting them costs one pass over the weights at load and saves an
	// offset calculation in the inner loop of every token.
	router []float32 // [hidden, experts]

	// expertGate, expertUp, and expertDown are indexed by expert id. Each entry is one
	// expert's slice of the corresponding stacked tensor.
	expertGate [][]float32 // [hidden, ffn] each
	expertUp   [][]float32 // [hidden, ffn] each
	expertDown [][]float32 // [ffn, hidden] each
}

// Weights is a fully loaded Llama model.
type Weights struct {
	Config Config

	// TokenEmbedding is [hidden, vocab]: one embedding per token.
	TokenEmbedding []float32

	// OutputNorm is [hidden].
	OutputNorm []float32

	// Output is [hidden, vocab]. It is nil when the model ties embeddings, in
	// which case TokenEmbedding serves as the output projection.
	Output []float32

	Layers []layer
}

// LoadWeights reads every tensor the forward pass needs.
//
// All tensors are dequantized up front. That costs the model's float32 size in
// memory -- 4 bytes per weight instead of the quantized size -- which is the
// wrong trade for a 7B model but the right one for a learning engine that wants
// the forward pass to be about the math. Keeping the quantized bytes and
// decoding per matmul is the optimization to make later.
func LoadWeights(path string) (*Weights, error) {
	l, err := quant.OpenLoader(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		// The loader is only needed during this call; its tensors are copied
		// into plain slices by the time LoadWeights returns.
		_ = l.Close()
	}()

	cfg, err := LoadConfig(l.File())
	if err != nil {
		return nil, err
	}

	// The frequency factors are a tensor, not metadata, so they cannot be read by
	// LoadConfig. A model without one leaves the field nil, which the rope treats as "no
	// scaling" and is what every model before Llama 3.2 relies on.
	if _, ok := l.File().Tensor("rope_freqs.weight"); ok {
		freqs, err := l.Tensor("rope_freqs.weight")
		if err != nil {
			return nil, fmt.Errorf("llama: read rope_freqs: %w", err)
		}
		if want := cfg.RopeDim / 2; len(freqs) != want {
			return nil, fmt.Errorf("llama: rope_freqs has %d values, want %d for rope dim %d",
				len(freqs), want, cfg.RopeDim)
		}
		cfg.RopeFreqs = freqs
	}

	names := []string{"token_embd.weight", "output_norm.weight"}
	if !cfg.TiedEmbeddings {
		names = append(names, "output.weight")
	}
	for i := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", i)
		names = append(names,
			p+"attn_norm.weight", p+"attn_q.weight", p+"attn_k.weight",
			p+"attn_v.weight", p+"attn_output.weight", p+"ffn_norm.weight",
		)
		if cfg.Experts > 0 {
			names = append(names,
				p+"ffn_gate_inp.weight", p+"ffn_gate_exps.weight",
				p+"ffn_up_exps.weight", p+"ffn_down_exps.weight",
			)
		} else {
			names = append(names,
				p+"ffn_gate.weight", p+"ffn_up.weight", p+"ffn_down.weight",
			)
		}
		if cfg.AttnBias {
			names = append(names, p+"attn_q.bias", p+"attn_k.bias", p+"attn_v.bias")
		}
	}

	// Decoding is bandwidth bound and the tensors are independent, so fan it
	// out. Four workers is enough to keep the disk and the integer decoding
	// units busy without oversubscribing memory bandwidth.
	tensors, err := l.Preload(names, 4)
	if err != nil {
		return nil, err
	}
	// A single non-finite weight does not stay local: it flows through the layer norms into
	// every logit, and the model then answers with whatever token 0 is -- tens of thousands of
	// perfectly valid ids that happen to spell nothing. The run exits 0, so nothing downstream
	// reports a problem, and the user concludes the model is bad rather than the file.
	//
	// Checking here costs one pass over the weights that were just decoded, against a load
	// that already read and dequantized them, and it turns that failure into a message that
	// names the tensor.
	if err := checkFinite(tensors); err != nil {
		return nil, err
	}

	w := &Weights{
		Config:         cfg,
		TokenEmbedding: tensors["token_embd.weight"],
		OutputNorm:     tensors["output_norm.weight"],
		Output:         tensors["output.weight"], // nil when tied
		Layers:         make([]layer, cfg.Layers),
	}
	for i := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", i)
		w.Layers[i] = layer{
			attnNorm: tensors[p+"attn_norm.weight"],
			wq:       tensors[p+"attn_q.weight"],
			wk:       tensors[p+"attn_k.weight"],
			wv:       tensors[p+"attn_v.weight"],
			wo:       tensors[p+"attn_output.weight"],
			ffnNorm:  tensors[p+"ffn_norm.weight"],
			qBias:    tensors[p+"attn_q.bias"],
			kBias:    tensors[p+"attn_k.bias"],
			vBias:    tensors[p+"attn_v.bias"],
		}
		if cfg.Experts > 0 {
			w.Layers[i].router = tensors[p+"ffn_gate_inp.weight"]
			w.Layers[i].expertGate = splitExperts(tensors[p+"ffn_gate_exps.weight"], cfg.Experts)
			w.Layers[i].expertUp = splitExperts(tensors[p+"ffn_up_exps.weight"], cfg.Experts)
			w.Layers[i].expertDown = splitExperts(tensors[p+"ffn_down_exps.weight"], cfg.Experts)
		} else {
			w.Layers[i].wGate = tensors[p+"ffn_gate.weight"]
			w.Layers[i].wUp = tensors[p+"ffn_up.weight"]
			w.Layers[i].wDown = tensors[p+"ffn_down.weight"]
		}
	}
	return w, nil
}

// checkFinite reports the first tensor holding a NaN or an infinity.
//
// Names are sorted so the message is the same on every run rather than depending on map order,
// which matters because a user comparing two load attempts would otherwise see a different
// tensor named each time -- the difference between "this file is corrupt" and "these files are
// corrupt in different places".
func checkFinite(tensors map[string][]float32) error {
	for _, name := range slices.Sorted(maps.Keys(tensors)) {
		values := tensors[name]
		for i, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("llama: weight %q has a non-finite value %v at index %d of %d;"+
					" the file is corrupt or truncated", name, v, i, len(values))
			}
		}
	}
	return nil
}

// splitExperts cuts a stacked expert tensor into one contiguous slice per expert.
//
// GGUF lists dimensions fastest-first and stores the outermost one as a contiguous
// run, so a [in, out, experts] tensor is experts blocks of in*out floats. Splitting
// costs one pass at load time and leaves the forward pass with plain matrices: the
// alternative is an expert stride threaded through every matmul call.
func splitExperts(stacked []float32, experts int) [][]float32 {
	if experts <= 0 || len(stacked) == 0 {
		return nil
	}
	per := len(stacked) / experts
	if per*experts != len(stacked) {
		// A stacked tensor whose element count is not divisible by the expert count
		// means the metadata and the weights disagree; validate catches this at load.
		return nil
	}
	out := make([][]float32, experts)
	for e := range experts {
		out[e] = stacked[e*per : (e+1)*per]
	}
	return out
}

// Params returns the number of weights the model holds.
func (w *Weights) Params() int64 {
	n := int64(len(w.TokenEmbedding)) + int64(len(w.OutputNorm)) + int64(len(w.Output))
	for _, l := range w.Layers {
		n += int64(len(l.attnNorm) + len(l.wq) + len(l.wk) + len(l.wv) + len(l.wo))
		n += int64(len(l.qBias) + len(l.kBias) + len(l.vBias))
		n += int64(len(l.ffnNorm) + len(l.wGate) + len(l.wUp) + len(l.wDown))
	}
	return n
}
