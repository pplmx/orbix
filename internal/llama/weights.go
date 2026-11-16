package llama

import (
	"fmt"

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

	ffnNorm []float32 // [hidden]
	wGate   []float32 // [hidden, ffn]
	wUp     []float32 // [hidden, ffn]
	wDown   []float32 // [ffn, hidden]
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

	names := []string{"token_embd.weight", "output_norm.weight"}
	if !cfg.TiedEmbeddings {
		names = append(names, "output.weight")
	}
	for i := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", i)
		names = append(names,
			p+"attn_norm.weight", p+"attn_q.weight", p+"attn_k.weight",
			p+"attn_v.weight", p+"attn_output.weight", p+"ffn_norm.weight",
			p+"ffn_gate.weight", p+"ffn_up.weight", p+"ffn_down.weight",
		)
	}

	// Decoding is bandwidth bound and the tensors are independent, so fan it
	// out. Four workers is enough to keep the disk and the integer decoding
	// units busy without oversubscribing memory bandwidth.
	tensors, err := l.Preload(names, 4)
	if err != nil {
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
			wGate:    tensors[p+"ffn_gate.weight"],
			wUp:      tensors[p+"ffn_up.weight"],
			wDown:    tensors[p+"ffn_down.weight"],
		}
	}
	return w, nil
}

// Params returns the number of weights the model holds.
func (w *Weights) Params() int64 {
	n := int64(len(w.TokenEmbedding)) + int64(len(w.OutputNorm)) + int64(len(w.Output))
	for _, l := range w.Layers {
		n += int64(len(l.attnNorm) + len(l.wq) + len(l.wk) + len(l.wv) + len(l.wo))
		n += int64(len(l.ffnNorm) + len(l.wGate) + len(l.wUp) + len(l.wDown))
	}
	return n
}
