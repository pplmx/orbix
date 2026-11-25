package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/tokenize"
)

// cmdRun implements "orbix run": load a model, generate, print the text.
func cmdRun(args []string) error {
	return runTo(os.Stdout, args)
}

// runTo is cmdRun with the destination as a parameter.
//
// The generated text is written as it arrives, which is behaviour a test has to observe while
// the run is still going: whether an interrupted run left anything behind is the property the
// streaming exists for, and it cannot be seen from the returned string alone. A parameter
// rather than os.Stdout, following chatSession, so a failing or recording writer is reachable.
func runTo(dst io.Writer, args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	model := modelFlag(fs)
	maxTokens := maxTokensFlag(fs, 128)
	temp := fs.Float64("temp", 0, "sampling temperature; 0 means greedy")
	topK := fs.Int("top-k", 40, "top-k sampling cutoff, 0 disables it. Only applies with --temp > 0")
	topP := fs.Float64("top-p", 0, "nucleus sampling cutoff in (0,1], 0 disables it. Only applies with --temp > 0")
	seed := fs.Int64("seed", 0, "sampling seed; 0 uses the current time. Has no effect without --temp, because greedy decoding is already deterministic")
	contextLen := fs.Int("ctx", 0, "context length; 0 uses the model's trained amount")
	echo := fs.Bool("echo", false, "print the prompt before the generated text")
	workers := workersFlag(fs)
	stats := fs.Bool("stats", false, "print timing and token counts to stderr")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := resolveModel(fs, model); err != nil {
		return err
	}
	prompt, err := promptFromArgs(fs, args)
	if err != nil {
		return err
	}
	if err := validateRunFlags(runFlags{
		temp:       *temp,
		topK:       *topK,
		topP:       *topP,
		workers:    *workers,
		maxTokens:  *maxTokens,
		contextLen: *contextLen,
	}); err != nil {
		return err
	}
	// Open the file twice: once for metadata and the tokenizer, once for the
	// weights. The directory parse keeps the two concerns from sharing a handle
	// that the weight loader closes -- but it is not cheap, which is why the load
	// timer starts before it. Measured on the 553 MiB Llama 3.2 file: gguf.Open
	// alone takes 1.47 s, and a report that began timing at NewWithOptions below
	// left that out of "load:" while wall-clock showed it.
	t0 := time.Now()
	gf, err := gguf.Open(*model)
	if err != nil {
		return err
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		return err
	}

	m, err := llama.NewWithOptions(*model, llama.Options{Workers: *workers})
	if err != nil {
		return err
	}
	loadTime := time.Since(t0)
	cfg := m.Config()

	nCtx := *contextLen
	if nCtx <= 0 {
		nCtx = cfg.ContextLen
	}
	// The rope table is built from the model's trained context, so a larger cache would ask for
	// positions the table does not hold -- which surfaced as "position 128 outside the rope table"
	// from inside the decode loop, after the prompt had already been processed.
	//
	// Extending the window is what rope scaling is for, and the scheme this engine applies is
	// whatever the file specifies: rope.scale_linear is read into RopeScale, and any other
	// rope.scaling.type is refused at load. There is no flag that adds scaling at run time, so a
	// checkpoint without it cannot be stretched by --ctx, and the answer is a refusal rather than
	// a cache that cannot be used.
	if nCtx > cfg.ContextLen {
		return fmt.Errorf("run: --ctx %d is past this model's trained context of %d;"+
			" going beyond it needs rope scaling in the model file, and this one does not carry it",
			nCtx, cfg.ContextLen)
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
	// A budget larger than what the window leaves is cut to fit, and saying so here rather than
	// only in the --stats report afterwards is the difference between a run the caller can
	// interrupt and one they wait out. Measured before this: `--max-tokens 40000` against a
	// 32768-position window sat generating for over two minutes with nothing on screen -- and
	// now that text streams, the caller does see progress, but still has no way to know the
	// number they typed was replaced.
	//
	// The note states the count that will actually be produced, which is the window minus the
	// prompt; --max-tokens is a cap, so a run that ends early on an end-of-sequence token still
	// produces fewer. It goes to stderr, so piped stdout continues to carry only the reply.
	if room := nCtx - len(promptTokens); *maxTokens > room {
		fmt.Fprintf(os.Stderr,
			"orbix: note: --max-tokens %d is past this model's %d-token window, so at most %d "+
				"tokens can be generated; lower --max-tokens to hear about a cut reply sooner\n",
			*maxTokens, nCtx, room)
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
			TopP:      float32(*topP),
			NextFloat: s.Float64,
		}
	}

	if *echo {
		// The prompt is part of the output, so its write failure is reported the same way the
		// streamed tokens' are rather than discarded: `orbix run --echo ... | head -1` breaks
		// the pipe at exactly this write, and a run that ignored it would go on generating a
		// reply nobody will read.
		if _, werr := io.WriteString(dst, prompt); werr != nil {
			return fmt.Errorf("run: write prompt: %w", werr)
		}
	}

	promptLen := len(promptTokens)
	out, genTime, err := generateStreaming(dst, tok, m, cache, promptTokens, sampler, *maxTokens)
	if err != nil {
		return err
	}
	// Decode only the generated part: the prompt is already on screen when --echo is set, and
	// repeating it otherwise would be confusing. Only the count is used now that the text has
	// been streamed, but the slice keeps the stats and the window note below reading the same
	// quantity they always did.
	generated := out[promptLen:]

	if *stats {
		n := len(generated)
		fmt.Fprintf(os.Stderr,
			"\nmodel:  %s\nparams: %d\nload:   %v\nprompt: %d tokens\n"+
				"gen:    %d tokens in %v (%.1f tok/s)\ncache:  %d/%d positions, %s reserved\n",
			cfg, m.W.Params(), loadTime.Round(time.Millisecond), promptLen,
			n, genTime.Round(time.Millisecond), float64(n)/genTime.Seconds(),
			cache.Len(), cache.Cap(), humanBytes(int64(cache.Bytes())))
		// A budget larger than the window is silently cut down to fit, which is the useful
		// behaviour -- but a run that asked for 128 and produced 123 looks like the model
		// stopped early, so say when the window is what stopped it. The test is the cache
		// being full rather than n < *maxTokens: a model that emits its end-of-sequence token
		// also produces fewer tokens than asked, and that is not a capacity limit.
		if cache.Len() == cache.Cap() && n < *maxTokens {
			// --ctx cannot help here: the window is the model's trained context, and a larger
			// cache would ask the rope table for positions it does not hold. The budget is what a
			// caller can lower, and the prompt is what they can shorten.
			fmt.Fprintf(os.Stderr,
				"note:   the %d-token window is full, so the %d-token budget was cut to %d;"+
					" shorten the prompt or lower --max-tokens\n",
				cache.Cap(), *maxTokens, n)
		}
	}
	return nil
}

// generateStreaming runs generation, writing each token's text to dst as it is produced.
//
// The text goes out as it arrives rather than after the whole generation finishes, so a reply
// that takes a minute shows progress and a reply the user interrupts keeps what had already been
// generated. Previously the run collected every token and printed once: measured on a 200-token
// reply against Qwen2.5-0.5B, stdout held 0 bytes after 8 seconds and 1096 at the end, so a
// SIGTERM lost the reply and a slow run was indistinguishable from a hang.
//
// Decoding goes through tokenize.StreamDecoder rather than DecodeCompletion for the reasons that
// type documents: decoding one token at a time with Decode strips the space before every word
// (turning "Once upon" into "Onceupon") and renders a character split across tokens as U+FFFD.
// Flush releases a partial character once generation ends.
//
// The returned duration covers generation only, which is what --stats reports; the caller
// measures load separately.
func generateStreaming(
	dst io.Writer,
	tok *tokenize.Tokenizer,
	m *llama.Model,
	cache llama.Cache,
	prompt []int32,
	sampler llama.Sampler,
	maxTokens int,
) ([]int32, time.Duration, error) {
	start := time.Now()
	dec := tok.Seq()
	out, err := m.Generate(cache, prompt, sampler, llama.GenerateOptions{
		MaxTokens: maxTokens,
		OnToken: func(token int32) error {
			if _, werr := io.WriteString(dst, dec.Token(token)); werr != nil {
				return fmt.Errorf("run: write output: %w", werr)
			}
			return nil
		},
	})
	if err != nil {
		return nil, 0, err
	}
	// The flush and the trailing newline go out even when generation produced nothing, so the
	// shell prompt returns on its own line either way.
	if _, werr := io.WriteString(dst, dec.Flush()+"\n"); werr != nil {
		return nil, 0, fmt.Errorf("run: write output: %w", werr)
	}
	return out, time.Since(start), nil
}

// promptFromArgs assembles the prompt from everything after the flags.
//
// Go's flag package stops at the first non-flag, so `orbix run "Once upon a time" --stats`
// parsed no flags at all: the run produced 123 tokens where 8 were asked for and printed no
// stats, and the flag after the prompt was not merely ignored but silently so. `model
// inspect` already refuses that shape, and the same rule belongs here -- except that run has
// to say what was wrong rather than count arguments, because a prompt legitimately arrives
// as more than one word.
//
// The words are joined rather than taking args[0] alone, which is the second half of the
// bug: with a prompt past the flags, `orbix run Once upon a time` prefilled the single token
// "Once" (2 tokens including BOS) and threw the rest away. Joining a single quoted word
// changes nothing, so the documented usage keeps working.
//
// args is the untruncated argument list, because fs.Args() cannot answer the question that
// matters here: whether the caller used "--". flag.Parse consumes the separator and hands
// back what follows it, so a prompt of "-5 degrees outside" arrives as an ordinary
// argument -- and was then refused, which made "--" useless for the one thing it exists for.
// Measured: `orbix run --model m -- "-5 degrees outside"` reported that flags must come
// before the prompt. With the separator present, no word is treated as a flag.
func promptFromArgs(fs *flag.FlagSet, args []string) (string, error) {
	words := fs.Args()
	if len(words) == 0 {
		return "", fmt.Errorf("run: missing prompt")
	}
	// Whether flag.Parse already consumed the separator changes what fs.Args() means.
	// If "--" is still in it, Parse stopped earlier, at a non-flag word, so everything after
	// the separator is a prompt word and everything before it must not be a flag.
	// If "--" is gone from fs.Args() but present in args, Parse consumed it after reading the
	// flags and all remaining words are prompt operands.
	// If there is no separator at all, every word is checked for a misplaced flag.
	at := slices.Index(words, "--")
	check := words
	switch {
	case at >= 0:
		check = words[:at]
		words = slices.Concat(words[:at], words[at+1:])
	case slices.Contains(args, "--"):
		check = nil
	}
	for _, w := range check {
		// Only a leading dash is treated as a flag: a prompt may contain "-" anywhere else,
		// and "the word starts a flag" is exactly what flag.Parse would have decided too.
		if strings.HasPrefix(w, "-") {
			return "", fmt.Errorf("run: flags must come before the prompt, got %q after it", w)
		}
	}
	// A lone "--" leaves no words, which is the same as typing nothing.
	if len(words) == 0 {
		return "", fmt.Errorf("run: missing prompt")
	}
	return strings.Join(words, " "), nil
}

// runFlags is what cmdRun read from its flags, before validation.
type runFlags struct {
	temp       float64
	topK       int
	topP       float64
	workers    int
	maxTokens  int
	contextLen int
}

// validateRunFlags refuses every flag value run cannot honour.
//
// The two kinds of check here read differently and both matter. Values the sampler compares
// against "> 0" are refused when negative, because a negative one silently means the same as
// zero -- greedy decoding and a top-k never applied -- which is the opposite of what "turn
// sampling down" reads as. Values that are NaN or infinite pass every range comparison, so each
// is named rather than left to one: a NaN temperature reached the sampler, made sum NaN, failed
// every `r < acc`, and returned the last candidate, the least likely token in the set.
//
// Checked before the model is opened, so a typo costs nothing.
func validateRunFlags(v runFlags) error {
	// Both of these are compared with "> 0" to decide whether sampling applies at all, so a
	// negative value silently means the same as zero: greedy decoding, and a top-k that is
	// never applied. A flag that reads as "turn sampling down" and turns it off is worth
	// refusing rather than interpreting.
	// NaN and Inf pass every comparison below, so each is named rather than left to a range
	// check: a NaN temperature reached the sampler, made sum NaN, failed every `r < acc`
	// and returned the last candidate -- the least likely token in the set.
	if math.IsNaN(v.temp) || math.IsInf(v.temp, 0) || v.temp < 0 {
		return fmt.Errorf("run: --temp must be a finite, non-negative number (0 means greedy)")
	}
	if v.workers < 0 {
		return fmt.Errorf("run: --workers must not be negative")
	}
	if v.topK < 0 {
		return fmt.Errorf("run: --top-k must not be negative (0 disables it)")
	}
	// A probability outside [0, 1] is refused rather than clamped, for the same reason as
	// --top-k. Zero and one both mean "no limit" to the sampler, and the flag documents 0 as
	// the value that disables the cutoff, so accepting them is the point rather than a
	// guess. What is refused is a value with no reading at all: clamping -0.1 up to 0 would
	// silently turn a typo into "off", and lowering 1.5 would apply a cutoff the caller did
	// not ask for.
	if math.IsNaN(v.topP) || v.topP < 0 || v.topP > 1 {
		return fmt.Errorf("run: --top-p must be between 0 and 1 (0 disables it)")
	}
	// The library rejects this too, but only once the model is loaded and Generate is
	// reached. Reporting it here means a typo costs nothing and does not depend on which
	// generation path the command happens to use.
	if v.maxTokens < 0 {
		return fmt.Errorf("run: --max-tokens must not be negative")
	}
	// Zero means "use the model's trained context", and the code reaches it with "<= 0", so
	// a negative length asked for the default and got it without comment. A context length
	// is the size of the arena being allocated, which is the last place to guess.
	if v.contextLen < 0 {
		return fmt.Errorf("run: --ctx must not be negative (0 uses the model's trained amount)")
	}

	return nil
}
