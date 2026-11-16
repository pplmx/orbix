package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/tokenize"
)

// cmdBench implements "orbix bench": measure generation throughput.
//
// The number that matters for a serving engine is tokens per second under
// concurrency, not single-stream latency. Reporting both is what shows whether
// interleaving requests buys anything on a given machine.
func cmdBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	model := fs.String("model", "", "path to a GGUF model file (required)")
	maxTokens := fs.Int("max-tokens", 32, "tokens to generate per request")
	concurrency := fs.Int("concurrency", 1, "requests in flight at once")
	workers := fs.Int("workers", 0, "kernel worker goroutines; 0 means one per CPU")
	requests := fs.Int("requests", 0, "total requests; 0 means one per concurrency slot")
	prompt := fs.String("prompt", "Once upon a time", "prompt for each request")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *model == "" {
		if v := os.Getenv("ORBIX_MODEL"); v != "" {
			*model = v
		} else {
			return fmt.Errorf("bench: --model is required (or set ORBIX_MODEL)")
		}
	}
	if *concurrency < 1 {
		return fmt.Errorf("bench: --concurrency must be at least 1")
	}
	if *maxTokens < 1 {
		return fmt.Errorf("bench: --max-tokens must be at least 1")
	}
	if *workers < 0 {
		return fmt.Errorf("bench: --workers must not be negative")
	}
	total := *requests
	if total <= 0 {
		total = *concurrency
	}

	gf, err := gguf.Open(*model)
	if err != nil {
		return err
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		return err
	}
	eng, err := engine.New(*model, engine.Options{MaxActive: *concurrency, Workers: *workers})
	if err != nil {
		return err
	}
	cfg := eng.Model().Config()
	// BOS comes from the model's own ids: a model that defines none must be
	// encoded without one, which is why this happens after loading the config.
	promptTokens := tok.Encode(*prompt, cfg.BOSID >= 0)
	if len(promptTokens) >= cfg.ContextLen {
		return fmt.Errorf("bench: prompt is %d tokens, context is %d", len(promptTokens), cfg.ContextLen)
	}

	fmt.Printf("model:       %s\n", cfg)
	fmt.Printf("params:      %d\n", eng.Model().W.Params())
	fmt.Printf("prompt:      %d tokens\n", len(promptTokens))
	fmt.Printf("generating:  %d tokens x %d requests, concurrency %d, workers %d\n\n",
		*maxTokens, total, *concurrency, eng.Model().Runner().Workers())

	start := time.Now()
	var generated int
	reason := map[string]int{}

	for done := 0; done < total; {
		// Submit up to the concurrency bound, then pump until something finishes.
		batch := min(*concurrency, total-done)
		chans := make([]<-chan engine.Event, batch)
		for i := range batch {
			ch, err := eng.Submit(engine.Request{
				Prompt:    promptTokens,
				MaxTokens: *maxTokens,
				Sampler:   llama.Greedy{},
			})
			if err != nil {
				return fmt.Errorf("bench: submit: %w", err)
			}
			chans[i] = ch
		}
		for eng.Pump() > 0 {
		}
		for _, ch := range chans {
			for ev := range ch {
				if ev.Err != nil {
					return fmt.Errorf("bench: %w", ev.Err)
				}
				if ev.HasToken {
					generated++
				}
				if ev.Done {
					reason[ev.FinishReason]++
				}
			}
		}
		done += batch
	}

	elapsed := time.Since(start)
	perSec := float64(generated) / elapsed.Seconds()
	fmt.Printf("generated:   %d tokens\n", generated)
	fmt.Printf("elapsed:     %v\n", elapsed.Round(time.Millisecond))
	fmt.Printf("throughput:  %.1f tok/s\n", perSec)
	fmt.Printf("per stream:  %.1f tok/s\n", perSec/float64(*concurrency))
	if len(reason) > 0 {
		fmt.Printf("finish:      %v\n", reason)
	}
	return nil
}
