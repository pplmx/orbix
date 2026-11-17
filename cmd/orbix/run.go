package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/tokenize"
)

// cmdRun implements "orbix run": load a model, generate, print the text.
func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	model := modelFlag(fs)
	maxTokens := maxTokensFlag(fs, 128)
	temp := fs.Float64("temp", 0, "sampling temperature; 0 means greedy")
	topK := fs.Int("top-k", 40, "top-k sampling cutoff, 0 disables it")
	seed := fs.Int64("seed", 0, "sampling seed; 0 uses the current time")
	contextLen := fs.Int("ctx", 0, "context length; 0 uses the model's trained amount")
	echo := fs.Bool("echo", false, "print the prompt before the generated text")
	stats := fs.Bool("stats", false, "print timing and token counts to stderr")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := resolveModel(fs, model); err != nil {
		return err
	}
	prompt := fs.Arg(0)
	if prompt == "" {
		return fmt.Errorf("run: missing prompt")
	}
	// Both of these are compared with "> 0" to decide whether sampling applies at all, so a
	// negative value silently means the same as zero: greedy decoding, and a top-k that is
	// never applied. A flag that reads as "turn sampling down" and turns it off is worth
	// refusing rather than interpreting.
	if *temp < 0 {
		return fmt.Errorf("run: --temp must not be negative (0 means greedy)")
	}
	if *topK < 0 {
		return fmt.Errorf("run: --top-k must not be negative (0 disables it)")
	}
	// The library rejects this too, but only once the model is loaded and Generate is
	// reached. Reporting it here means a typo costs nothing and does not depend on which
	// generation path the command happens to use.
	if *maxTokens < 0 {
		return fmt.Errorf("run: --max-tokens must not be negative")
	}
	// Zero means "use the model's trained context", and the code reaches it with "<= 0", so
	// a negative length asked for the default and got it without comment. A context length
	// is the size of the arena being allocated, which is the last place to guess.
	if *contextLen < 0 {
		return fmt.Errorf("run: --ctx must not be negative (0 uses the model's trained amount)")
	}

	// Open the file twice: once for metadata and the tokenizer, once for the
	// weights. The directory parse is cheap and keeps the two concerns from
	// sharing a handle that the weight loader closes.
	gf, err := gguf.Open(*model)
	if err != nil {
		return err
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		return err
	}

	t0 := time.Now()
	m, err := llama.New(*model)
	if err != nil {
		return err
	}
	loadTime := time.Since(t0)
	cfg := m.Config()

	nCtx := *contextLen
	if nCtx <= 0 {
		nCtx = cfg.ContextLen
	}
	cache, err := llama.NewKVCache(cfg, nCtx)
	if err != nil {
		return err
	}

	// The tokenizer's own special-token handling decides whether BOS is
	// prepended; a model without a BOS id encodes the prompt as-is.
	promptTokens := tok.Encode(prompt, cfg.BOSID >= 0)
	if len(promptTokens) == 0 {
		return fmt.Errorf("run: prompt encoded to no tokens")
	}
	if len(promptTokens) >= nCtx {
		return fmt.Errorf("run: prompt is %d tokens, context is %d", len(promptTokens), nCtx)
	}

	var sampler llama.Sampler = llama.Greedy{}
	if *temp > 0 {
		s := rand.New(rand.NewSource(*seed))
		if *seed == 0 {
			s = rand.New(rand.NewSource(time.Now().UnixNano()))
		}
		sampler = llama.Temperature{
			Temp:      float32(*temp),
			TopK:      *topK,
			NextFloat: s.Float64,
		}
	}

	if *echo {
		fmt.Print(prompt)
	}

	tGen := time.Now()
	promptLen := len(promptTokens)
	out, err := m.Generate(cache, promptTokens, sampler, llama.GenerateOptions{
		MaxTokens: *maxTokens,
	})
	if err != nil {
		return err
	}
	genTime := time.Since(tGen)

	// Decode only the generated part: the prompt is already on screen when
	// --echo is set, and repeating it otherwise would be confusing.
	generated := out[promptLen:]
	fmt.Println(tok.DecodeCompletion(generated))

	if *stats {
		n := len(generated)
		fmt.Fprintf(os.Stderr,
			"\nmodel:  %s\nparams: %d\nload:   %v\nprompt: %d tokens\n"+
				"gen:    %d tokens in %v (%.1f tok/s)\ncache:  %d/%d positions, %d bytes\n",
			cfg, m.W.Params(), loadTime.Round(time.Millisecond), promptLen,
			n, genTime.Round(time.Millisecond), float64(n)/genTime.Seconds(),
			cache.Len(), cache.Cap(), cache.Bytes())
	}
	return nil
}
