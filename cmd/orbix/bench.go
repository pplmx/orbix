package main

import (
	"flag"
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"
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
	// Zero has a documented meaning here -- one request per concurrency slot -- and the code
	// reaches it with "total <= 0", so a negative value was silently the same as zero. The
	// difference matters in a benchmark: the number of requests is the sample size, and a
	// mistyped one is not a smaller measurement but a different one.
	if *requests < 0 {
		return fmt.Errorf("bench: --requests must not be negative (0 means one per " +
			"concurrency slot)")
	}
	// The suite fixes its own concurrency, request count, admission policy, and cache shape:
	// those are the variables the rows compare, so letting a flag override them would remove
	// the comparison the table exists to make. Which is exactly why the flags cannot be
	// accepted and left unread. Passing `--suite --policy bogus` used to run the whole table
	// and exit 0 with the invalid value never looked at, and `--suite --concurrency 32` printed
	// a table of eight scenarios measured at other concurrencies with nothing to say so.
	//
	// Checked by walking what was actually set rather than by comparing against defaults: a
	// default is indistinguishable from an explicit equal value, and the point is to tell the
	// operator their flag did nothing.
	if *suite {
		var ignored []string
		usedBySuite := map[string]bool{
			"model": true, "max-tokens": true, "prompt": true, "workers": true, "suite": true,
		}
		fs.Visit(func(f *flag.Flag) {
			if !usedBySuite[f.Name] {
				ignored = append(ignored, "--"+f.Name)
			}
		})
		if len(ignored) > 0 {
			return fmt.Errorf("bench: --suite measures a fixed set of scenarios and does not use %s;"+
				" drop the flag, or run without --suite to measure one configuration",
				strings.Join(ignored, ", "))
		}
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

	// Two settings only take effect with a pool, and both used to be accepted and then
	// ignored: the run stayed on the contiguous cache while the report still described what
	// the operator had asked for. --prefix-cache was worse than ignored -- the reuse note at
	// the end of the run dereferenced the pool that --blocks would have created, so the run
	// measured everything and then panicked on its last print instead of reporting.
	//
	// Checked before the model is opened, so a mistyped run costs nothing. The suite above is
	// exempt because it configures its own cache for each scenario and uses neither flag.
	if *prefixCache && *cache.blocks == 0 {
		return fmt.Errorf("bench: --prefix-cache needs a paged cache; add --blocks (and " +
			"optionally --block-size), or drop --prefix-cache")
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
		// Sorted and spelled out rather than printed as a map: this is the line a reader uses to
		// tell a truncated run from a complete one, and "map[length:8 stop:4]" makes them parse Go
		// syntax to find out. Sorted because map iteration order is random, and two runs of the same
		// benchmark should print the same report.
		parts := make([]string, 0, len(reason))
		for _, k := range slices.Sorted(maps.Keys(reason)) {
			parts = append(parts, fmt.Sprintf("%s=%d", k, reason[k]))
		}
		fmt.Printf("finish:      %s\n", strings.Join(parts, " "))
	}
	// Only when the flag was on, and only when there is something to report: a "0 reused" line
	// on a run that could not have reused anything would read as a defect rather than as a
	// measurement. The suite already prints this for its reuse pair; a one-off bench with
	// --prefix-cache was the one place an operator could turn sharing on and get no feedback
	// about whether it did anything.
	if *prefixCache {
		if reused := eng.Stats().PrefixReused; reused > 0 {
			fmt.Printf("reused:      %d prompt positions\n", reused)
		} else {
			// Concurrent requests prefill together, so none of them can attach a prefix another
			// has not written yet. Saying so is the difference between "sharing is broken" and
			// "sharing cannot apply to this shape of run" -- the reuse pair in --suite is
			// sequential for exactly this reason.
			// Reuse works at whole blocks, so the default five-token prompt has nothing to
			// share however many requests run. Measured on this model: a prompt that spans a
			// 16-position block reports reuse at both concurrency 1 and concurrency 16, so
			// concurrency is not the variable -- prompt length is.
			fmt.Printf("reused:      none (a prompt shorter than one %d-position block leaves no\n"+
				"             whole block to share, whatever the concurrency)\n", eng.Pool().BlockSize())
		}
	}
	return nil
}
