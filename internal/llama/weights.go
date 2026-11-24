package llama

import (
	"fmt"
	"maps"
	"math"
	"slices"

	"github.com/pplmx/orbix/internal/kernel"
	"github.com/pplmx/orbix/internal/quant"
)

// layer is the weight set of one transformer block.
//
// Weights are stored dequantized, in ggml's row-major layout: a projection is a
// slice of out rows, each holding in contiguous weights. That is the orientation
// the kernels expect, so the forward pass never transposes anything.
type layer struct {
	attnNorm []float32 // [hidden]
	// attnPostNorm and ffnPostNorm are [hidden], and nil on architectures that add sub-layer
	// outputs to the residual without normalizing them first.
	attnPostNorm []float32
	ffnPostNorm  []float32
	wq           []float32 // [hidden, heads*headDim]
	wk           []float32 // [hidden, kvHeads*headDim]
	wv           []float32 // [hidden, kvHeads*headDim]
	wo           []float32 // [heads*headDim, hidden]

	// qBias, kBias, and vBias are added to the projections before attention. They
	// are nil for architectures that do not use them (Llama) and present for those
	// that do (Qwen2). Keeping them separate from the weights rather than folding
	// them into the matrices is what makes a nil check the only branch the forward
	// pass needs.
	qBias []float32 // [heads*headDim]
	kBias []float32 // [kvHeads*headDim]
	vBias []float32 // [kvHeads*headDim]

	ffnNorm []float32 // [hidden]
	// qNorm and kNorm are [head dim]: one vector per layer, applied to every head's query and
	// key before the rotation. Nil on architectures that do not normalise them, which is why
	// the forward pass branches on Config.QKNorm rather than on these being present.
	qNorm []float32
	kNorm []float32
	wGate []float32 // [hidden, ffn]
	wUp   []float32 // [hidden, ffn]
	wDown []float32 // [ffn, hidden]

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
		// This tensor is loaded outside the batch below, so it needs the same check the
		// weights get: a NaN frequency factor poisons the rotation for every position, which
		// is the same silent-garbage failure the batch guard exists to prevent.
		if err := checkFinite(map[string][]float32{"rope_freqs.weight": freqs}); err != nil {
			return nil, err
		}
		// Zero is finite, so checkFinite passes it, and the rope divides this dimension's
		// theta by it: theta/0 is +Inf, math.Cos(+Inf) is NaN, and every position's cos and
		// sin for that dimension pair are NaN from the first token on. Measured with
		// freqs {1, 0}: cos[pos=1][1] is NaN. It is the failure the check above exists to
		// prevent, reached by a value that checks as finite.
		if i := slices.Index(freqs, 0); i >= 0 {
			return nil, fmt.Errorf("llama: rope_freqs.weight has a zero at index %d of %d;"+
				" a zero factor divides the rotation angle by zero, so every position's"+
				" cos and sin become NaN -- the file is corrupt", i, len(freqs))
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
		if cfg.PostNorms {
			names = append(names, p+"post_attention_norm.weight", p+"post_ffw_norm.weight")
		}
		if cfg.QKNorm {
			names = append(names, p+"attn_q_norm.weight", p+"attn_k_norm.weight")
		}
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
			attnNorm:     tensors[p+"attn_norm.weight"],
			wq:           tensors[p+"attn_q.weight"],
			wk:           tensors[p+"attn_k.weight"],
			wv:           tensors[p+"attn_v.weight"],
			wo:           tensors[p+"attn_output.weight"],
			ffnNorm:      tensors[p+"ffn_norm.weight"],
			qNorm:        tensors[p+"attn_q_norm.weight"],
			kNorm:        tensors[p+"attn_k_norm.weight"],
			attnPostNorm: tensors[p+"post_attention_norm.weight"],
			ffnPostNorm:  tensors[p+"post_ffw_norm.weight"],
			qBias:        tensors[p+"attn_q.bias"],
			kBias:        tensors[p+"attn_k.bias"],
			vBias:        tensors[p+"attn_v.bias"],
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
// embedInto copies one token's embedding row into dst, applying the architecture's input scale.
//
// The scale is applied here rather than folded into the matrix because the same matrix is
// usually the tied output projection, and that has to see the unscaled weights. For every
// architecture but Gemma the factor is 1 and the multiply is the copy.
func embedInto(dst, row []float32, scale float32) {
	if scale == 1 {
		copy(dst, row)
		return
	}
	for i, v := range row {
		dst[i] = v * scale
	}
}

// embedRow copies one token's embedding into dst, checking the bounds first.
//
// The three forward paths -- single token, batched prefill, batched decode -- each had their own
// copy of this check, and their own comment apologising for it: "this is the third copy of the
// check and it was missed when the first two were corrected". The comment was right about the
// cause and the copies were right about the rule; what was missing is that the rule has one
// home.
//
// The bound is on the token rather than on the computed offset, and that is the whole point. A
// check written as `off+Hidden > len(embedding)` cannot see a negative token: off becomes
// negative, off+Hidden stays below the slice length, and the copy that follows panics on a slice
// expression starting below zero. Computing the offset only after the token is known to be in
// range also keeps it from overflowing on a 32-bit int.
func (m *Model) embedRow(dst []float32, token int) error {
	cfg := m.W.Config
	if token < 0 || token >= cfg.VocabSize {
		return fmt.Errorf("llama: token %d outside vocabulary of %d", token, cfg.VocabSize)
	}
	off := token * cfg.Hidden
	if off+cfg.Hidden > len(m.W.TokenEmbedding) {
		return fmt.Errorf("llama: token %d embedding outside %d values", token, len(m.W.TokenEmbedding))
	}
	embedInto(dst, m.W.TokenEmbedding[off:off+cfg.Hidden], cfg.EmbedScale)
	return nil
}

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
//
// The expert tensors count too. A mixture-of-experts layer replaces the three dense matrices with
// one router and a stack of per-expert ones, so summing only wGate/wUp/wDown reports a few percent
// of the true figure for a model whose entire size is in the experts -- and this number is printed
// by `orbix run --stats` and `orbix bench`, which makes it a claim about the model rather than an
// internal counter.
func (w *Weights) Params() int64 {
	n := int64(len(w.TokenEmbedding)) + int64(len(w.OutputNorm)) + int64(len(w.Output))
	for _, l := range w.Layers {
		n += int64(len(l.attnNorm) + len(l.wq) + len(l.wk) + len(l.wv) + len(l.wo))
		n += int64(len(l.qBias) + len(l.kBias) + len(l.vBias))
		n += int64(len(l.ffnNorm) + len(l.wGate) + len(l.wUp) + len(l.wDown))
		n += int64(len(l.router))
		for _, e := range l.expertGate {
			n += int64(len(e))
		}
		for _, e := range l.expertUp {
			n += int64(len(e))
		}
		for _, e := range l.expertDown {
			n += int64(len(e))
		}
	}
	return n
}

// outputWeight returns the matrix the logits are projected through.
//
// A tied model reuses the embedding matrix, which has the same shape as an output projection:
// [hidden, vocab]. Three forward paths had this choice written out, and a fourth that forgot it
// would silently project through a nil slice or skip a real matrix.
func (m *Model) outputWeight() []float32 {
	if m.W.Output != nil {
		return m.W.Output
	}
	return m.W.TokenEmbedding
}

// capLogits applies the output softcap, if the architecture declares one.
//
// This is the single home for a step that exists in three paths -- Forward, Prefill and
// DecodeStep -- and was missed in one of them: docs/models.md records that Gemma 2's cap was
// added to the single-token and batched-decode projections but not the prefill's, so every
// reply's first token came from a different distribution than the rest. The defect was not the
// cap's arithmetic; it was that the arithmetic had three homes and one of them did not know.
//
// Taking the slice rather than reading a field is what makes one call serve all three: the
// prefill path has one row of logits, the decode path has n.
func (m *Model) capLogits(logits []float32) {
	kernel.Softcap(logits, m.W.Config.FinalSoftcap)
}
