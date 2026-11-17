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
//
// With --suite, it runs a fixed set of scenarios instead and prints them as one
// table. A single number answers one question about one machine; the suite is what
// shows which of the engine's mechanisms actually pay for themselves, and it is the
// baseline any later optimisation has to beat.
func cmdBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	model := modelFlag(fs)
	maxTokens := maxTokensFlag(fs, 32)
	concurrency := fs.Int("concurrency", 1, "requests in flight at once")
	cache := registerCacheFlags(fs, true)
	prefixCache := fs.Bool("prefix-cache", false, "reuse KV blocks across requests that share a prompt prefix")
	policy := fs.String("policy", "fifo", "admission policy when more requests are ready than --concurrency: fifo, shortest, or fair")
	requests := fs.Int("requests", 0, "total requests; 0 means one per concurrency slot")
	prompt := fs.String("prompt", "Once upon a time", "prompt for each request")
	suite := fs.Bool("suite", false, "run the fixed scenario suite instead of one measurement")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := resolveModel(fs, model); err != nil {
		return err
	}
	if *concurrency < 1 {
		return fmt.Errorf("bench: --concurrency must be at least 1")
	}
	if *maxTokens < 1 {
		return fmt.Errorf("bench: --max-tokens must be at least 1")
	}
	if err := cache.validate("bench"); err != nil {
		return err
	}
	total := *requests
	if total <= 0 {
		total = *concurrency
	}

	// The suite builds its own engines per scenario and encodes its own prompt, so it
	// branches before the single-measurement path constructs one it would not use.
	if *suite {
		tok, cfg, err := benchLoad(*model)
		if err != nil {
			return err
		}
		promptIDs := tok.Encode(*prompt, cfg.BOSID >= 0)
		if len(promptIDs) >= cfg.ContextLen {
			return fmt.Errorf("bench: prompt is %d tokens, context is %d", len(promptIDs), cfg.ContextLen)
		}
		return runSuite(suiteOptions{
			model:   *model,
			prompt:  promptIDs,
			maxTok:  *maxTokens,
			workers: *cache.workers,
		})
	}

	gf, err := gguf.Open(*model)
	if err != nil {
		return err
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		return err
	}
	schedPolicy, err := engine.ParsePriority(*policy)
	if err != nil {
		return err
	}
	eng, err := engine.New(*model, engine.Options{
		MaxActive:   *concurrency,
		Policy:      schedPolicy,
		Workers:     *cache.workers,
		BlockSize:   *cache.blockSize,
		Blocks:      *cache.blocks,
		PrefixCache: *prefixCache,
	})
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
	fmt.Printf("generating:  %d tokens x %d requests, concurrency %d, %s policy, workers %d\n",
		*maxTokens, total, *concurrency, schedPolicy, eng.Model().Runner().Workers())
	if pool := eng.Pool(); pool != nil {
		fmt.Printf("kv cache:    paged, %d blocks x %d positions = %.2f MiB shared\n",
			pool.Blocks(), pool.BlockSize(), float64(pool.Bytes())/(1<<20))
	} else {
		perSeq := int64(cfg.ContextLen) * int64(cfg.KVHeads) * int64(cfg.HeadDim) * 2 * 4 * int64(cfg.Layers)
		fmt.Printf("kv cache:    contiguous, %.2f MiB reserved per request\n",
			float64(perSeq)/(1<<20))
	}
	fmt.Println()

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
