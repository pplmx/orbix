package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/tokenize"
)

// cmdChat implements "orbix chat": an interactive REPL with conversation history.
//
// It runs on the engine rather than on llama directly, which is what makes the
// session cheap: every turn re-sends the history, and with a paged cache plus prefix
// reuse the engine attaches the blocks the previous turn already computed instead of
// prefilling them again. A REPL that called the model directly would pay the full
// prefill for the whole conversation on every turn, which is the difference between a
// usable session and one where each question takes longer than the last.
func cmdChat(args []string) error {
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	model := modelFlag(fs)
	system := fs.String("system", "", "system prompt prepended to the conversation")
	maxTokens := maxTokensFlag(fs, 128)
	temp := fs.Float64("temp", 0, "sampling temperature; 0 means greedy")
	topK := fs.Int("top-k", 40, "top-k sampling cutoff, 0 disables it")
	topP := fs.Float64("top-p", 0, "nucleus sampling cutoff in (0,1], 0 disables it")
	cache := registerCacheFlags(fs, true)
	noPrefix := fs.Bool("no-prefix-cache", false, "disable prefix reuse between turns")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := resolveModel(fs, model); err != nil {
		return err
	}
	if err := cache.validate("chat"); err != nil {
		return err
	}
	// Sampling is switched on by "temp > 0" and top-k applies at "topK > 0", so a negative
	// value is the same as zero: the session goes greedy without saying so.
	if *temp < 0 {
		return fmt.Errorf("chat: --temp must not be negative (0 means greedy)")
	}
	if *topK < 0 {
		return fmt.Errorf("chat: --top-k must not be negative (0 disables it)")
	}
	if *topP < 0 || *topP > 1 {
		return fmt.Errorf("chat: --top-p must be between 0 and 1 (0 disables it)")
	}
	// The engine refuses this too, but only when a turn is submitted -- so the user typed a
	// question, waited for the model to load, and then read an error about the flag. Checking
	// it here reports it before the session starts.
	if *maxTokens < 0 {
		return fmt.Errorf("chat: --max-tokens must not be negative")
	}

	return chatSession(os.Stdin, os.Stdout, chatOptions{
		model:       *model,
		system:      *system,
		maxTokens:   *maxTokens,
		temp:        *temp,
		topK:        *topK,
		topP:        *topP,
		workers:     *cache.workers,
		blockSize:   *cache.blockSize,
		blocks:      *cache.blocks,
		prefixCache: !*noPrefix,
	})
}

// chatOptions is what a session needs once the flags are parsed.
type chatOptions struct {
	model       string
	system      string
	maxTokens   int
	temp        float64
	topK        int
	topP        float64
	workers     int
	blockSize   int
	blocks      int
	prefixCache bool

	// renderer produces the prompt for a conversation. It is resolved once per
	// session, because compiling the model's template per turn would reparse the
	// same few hundred bytes on every message.
	renderer *engine.ChatRenderer
}

// chatSession runs the read-eval-print loop.
//
// in and out are parameters rather than os.Stdin and os.Stdout so a test can drive a
// whole conversation without a subprocess. The loop's contract is small on purpose:
// one line in, one reply out, and history that only grows when a turn succeeds.
func chatSession(in io.Reader, out io.Writer, opts chatOptions) error {
	gf, err := gguf.Open(opts.model)
	if err != nil {
		return err
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		return err
	}
	// The model's own chat format when this project can evaluate it, the neutral
	// transcript otherwise. Built once per session: compiling a template per turn
	// would reparse the same few hundred bytes every time.
	renderer, err := engine.NewChatRenderer(gf)
	if err != nil {
		return err
	}
	opts.renderer = renderer
	// Say which format is in use. It is worth one line: a model whose template was
	// not applied answers in a shape the caller did not expect, and knowing that from
	// the start is the difference between a bug report and a mystery.
	banner := "(using the model's own chat template)"
	if !renderer.UsesModelTemplate() {
		banner = fmt.Sprintf("(using a neutral transcript: %s)", renderer.FallbackReason())
	}
	if _, err := fmt.Fprintln(out, banner); err != nil {
		return err
	}

	// A paged cache with prefix reuse is the default for a REPL, and it is sized
	// from the model's own context: a conversation longer than the window cannot be
	// remembered anyway, so the pool only has to hold one.
	blockSize := opts.blockSize
	if blockSize <= 0 {
		blockSize = 16
	}
	// Size a paged pool for one full context unless the caller asked for something
	// else. A REPL is exactly the workload paging plus prefix reuse exists for: every
	// turn re-sends the history, so the pool must hold one conversation and the
	// prefix index must be able to hand back the previous turn's blocks.
	blocks := opts.blocks
	if blocks <= 0 {
		blocks = (cfgForBlocks(gf) + blockSize - 1) / blockSize
	}

	eng, err := engine.New(opts.model, engine.Options{
		MaxActive: 1, // a REPL has one request at a time
		Workers:   opts.workers,
		BlockSize: blockSize,
		Blocks:    blocks,
	})
	if err != nil {
		return err
	}
	if opts.prefixCache {
		if pool := eng.Pool(); pool != nil {
			pool.EnablePrefixCache()
		}
	}
	cfg := eng.Model().Config()

	if _, err := fmt.Fprintf(out, "orbix chat: %s\n", cfg); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(out, "Type a message and press enter. Commands: /exit, /clear, /history, /help\n\n"); err != nil {
		return err
	}

	var history []chatTurn
	// notedFallback records that the session has already explained a render-time fallback.
	notedFallback := false
	var systemTurn string
	if opts.system != "" {
		systemTurn = opts.system
	}

	scanner := bufio.NewScanner(in)
	// A long paste is a single line as far as the REPL is concerned, so the default
	// 64 KiB token limit is raised: it is a log line, not a chat message.
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for {
		if _, err := fmt.Fprint(out, "> "); err != nil {
			return err
		}
		if !scanner.Scan() {
			// EOF: a piped session ends the same way /exit does. A newline keeps the
			// shell prompt on its own line.
			_, _ = fmt.Fprintln(out)
			return scanner.Err()
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "/") {
			done, err := chatCommand(out, line, &history, eng.Stats)
			if err != nil {
				return err
			}
			if done {
				return nil
			}
			continue
		}

		reply, err := chatTurnOnce(out, eng, tok, systemTurn, history, line, opts)
		if err != nil {
			// A failed turn leaves the history untouched, so the next question is
			// asked against the same context rather than against a broken one.
			if _, werr := fmt.Fprintf(out, "error: %v\n", explainChatError(err)); werr != nil {
				return werr
			}
			continue
		}
		history = append(history,
			chatTurn{role: "user", text: line},
			chatTurn{role: "assistant", text: reply},
		)
		// The template can parse and still fail to render this conversation -- an
		// unsupported field on a message, for instance. That fallback happens inside the
		// renderer, so the banner printed above cannot know about it; say it once, the
		// first time it happens, rather than leaving the session silently on the neutral
		// transcript.
		// Only a template that parsed and then failed to render *this* conversation is news.
		// A model with no template was already explained by the banner, and checking only
		// UsesModelTemplate made the REPL announce a render-time fallback on the first
		// successful turn of every such session -- contradicting the banner it had just
		// printed, and describing a failure that never happened.
		if !notedFallback && opts.renderer != nil && opts.renderer.HasTemplate() &&
			!opts.renderer.UsesModelTemplate() {
			notedFallback = true
			if _, werr := fmt.Fprintf(out, "(the model's template could not render this "+
				"conversation: %s; using a neutral transcript)\n",
				opts.renderer.FallbackReason()); werr != nil {
				return werr
			}
		}
	}
}

// cfgForBlocks returns the context length the pool has to cover.
//
// The key is derived from general.architecture rather than listed per architecture, so
// adding a third one does not mean adding a line here -- a list would be a second
// place to forget.
func cfgForBlocks(f *gguf.File) int {
	arch, err := f.Arch()
	if err != nil {
		return defaultBlockContext
	}
	if v, ok := f.Get[int](arch + ".context_length"); ok && v > 0 {
		return v
	}
	return defaultBlockContext
}

// defaultBlockContext is the pool size used when a file does not state its context
// length. It matches llama's smallest common window rather than being zero, because a
// zero pool would silently disable paging.
const defaultBlockContext = 2048

// chatTurn is one exchanged message.
type chatTurn struct {
	role string
	text string
}

// chatTurnOnce sends one user message and streams the reply.
//
// The prompt is rebuilt from the whole history each turn, which is what a chat model
// expects and what makes prefix reuse pay: every turn's prompt starts with the
// previous turn's prompt, so the engine can attach those blocks instead of
// recomputing them.
// explainChatError adds what the caller can do about an error the engine reports bare.
//
// Exhausting the pool is the one failure here that repeats: the history is not advanced on a failed
// turn, so the next turn submits the same conversation against the same full pool and fails
// identically, for as long as the session lasts. The engine's message names the pool's size, which
// is the wrong end of the problem -- the session already knows what it asked for -- and suggests
// nothing.
//
// The pool is sized for one full context by default, so this only happens when --blocks was set
// below that. /clear is the way out that does not need a restart.
func explainChatError(err error) error {
	if !errors.Is(err, llama.ErrPoolExhausted) {
		return err
	}
	// The engine's message already gives the pool's size, so this adds only what it cannot know:
	// that the session will repeat the failure, and the two ways out.
	return fmt.Errorf("%w -- this conversation outgrew the pool, and the next turn will fail the same"+
		" way because a failed turn does not grow the history; /clear starts a new conversation,"+
		" or leave --blocks unset to size the pool for the whole context", err)
}

func chatTurnOnce(out io.Writer, eng *engine.Engine, tok *tokenize.Tokenizer, system string, history []chatTurn, user string, opts chatOptions) (string, error) {
	prompt, err := opts.renderer.Render(chatTurns(system, history, user))
	if err != nil {
		return "", err
	}
	cfg := eng.Model().Config()
	// The chat transcript is markup, not prose: its markers are control tokens the
	// model was trained to see, so they have to be matched whole.
	ids := tok.EncodeWithSpecials(prompt, cfg.BOSID >= 0)
	if len(ids) == 0 {
		return "", fmt.Errorf("prompt encoded to no tokens")
	}
	if len(ids) >= cfg.ContextLen {
		return "", fmt.Errorf("conversation is %d tokens, context is %d; use /clear", len(ids), cfg.ContextLen)
	}

	sampler := llama.Sampler(llama.Greedy{})
	if opts.temp > 0 {
		// math/rand's generator, not crypto/rand: sampling quality does not depend on
		// unpredictability, and a REPL benefits from the cheap path.
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		sampler = llama.Temperature{
			Temp: float32(opts.temp), TopK: opts.topK, TopP: float32(opts.topP),
			NextFloat: r.Float64,
		}
	}

	maxTokens := opts.maxTokens
	if room := cfg.ContextLen - len(ids); maxTokens > room {
		maxTokens = room
	}
	ch, err := eng.Submit(engine.Request{
		Prompt:    ids,
		MaxTokens: maxTokens,
		Sampler:   sampler,
	})
	if err != nil {
		return "", err
	}

	dec := tok.Seq()
	var reply strings.Builder
	// The engine advances sequences only inside Pump, so this loop drives the model
	// and drains the channel in one goroutine. Nothing else can make progress, which
	// is why a REPL needs no locks and no goroutine per token.
	//
	// Each round: emit whatever the channel already holds, then let the engine run
	// one scheduling round. Pump reports how many sequences are still active, so a
	// zero means the sequence finished this round and the remainder can be drained.
	for {
		if err := drainChatEvents(ch, out, dec, &reply); err != nil {
			return "", err
		}
		if eng.Pump() == 0 {
			// The final round may have emitted tokens before the sequence closed; the
			// channel is closed by now, so this drains to completion and returns.
			for ev := range ch {
				if ev.Err != nil {
					return "", ev.Err
				}
				if ev.HasToken {
					text := dec.Token(ev.Token)
					if _, err := fmt.Fprint(out, text); err != nil {
						return "", err
					}
					reply.WriteString(text)
				}
			}
			// A character split across the last tokens is still held by the decoder,
			// waiting for a byte that never arrives. Dropping it would make the printed
			// reply differ from the non-streaming decode of the same tokens.
			if tail := dec.Flush(); tail != "" {
				if _, err := fmt.Fprint(out, tail); err != nil {
					return "", err
				}
				reply.WriteString(tail)
			}
			if _, err := fmt.Fprintln(out); err != nil {
				return "", err
			}
			return reply.String(), nil
		}
	}
}

// drainChatEvents prints every event already queued, returning the sequence's error if
// it reported one.
//
// It must not block: more events only arrive while Pump runs, so waiting here would
// deadlock against the engine rather than make progress.
func drainChatEvents(ch <-chan engine.Event, out io.Writer, dec *tokenize.StreamDecoder, reply *strings.Builder) error {
	for {
		select {
		case ev, ok := <-ch:
			if !ok {
				// The channel closed, so no more tokens can complete a held-back character.
				if tail := dec.Flush(); tail != "" {
					if _, err := fmt.Fprint(out, tail); err != nil {
						return err
					}
					reply.WriteString(tail)
				}
				return nil
			}
			if ev.Err != nil {
				return ev.Err
			}
			if ev.HasToken {
				text := dec.Token(ev.Token)
				if _, err := fmt.Fprint(out, text); err != nil {
					return err
				}
				reply.WriteString(text)
			}
		default:
			return nil
		}
	}
}

// renderChatPrompt renders a conversation through the shared template, so a REPL
// session and an HTTP request over the same conversation produce the same tokens.
// chatTurns flattens a session's history into the turn list a renderer consumes.
func chatTurns(system string, history []chatTurn, user string) []engine.Turn {
	turns := make([]engine.Turn, 0, len(history)+2)
	if system != "" {
		turns = append(turns, engine.Turn{Role: "system", Text: system})
	}
	for _, turn := range history {
		turns = append(turns, engine.Turn{Role: turn.role, Text: turn.text})
	}
	turns = append(turns, engine.Turn{Role: "user", Text: user})
	return turns
}

// chatCommand handles a slash command, reporting whether the session should end.
//
// The write errors are propagated rather than ignored: a session whose terminal has
// gone away has nothing useful left to do, and silently continuing would generate
// tokens into a broken pipe.
func chatCommand(out io.Writer, line string, history *[]chatTurn, stats func() engine.Stats) (bool, error) {
	switch line {
	case "/exit", "/quit", "/q":
		return true, nil
	case "/clear":
		*history = nil
		_, err := fmt.Fprintln(out, "history cleared")
		return false, err
	case "/history":
		if len(*history) == 0 {
			_, err := fmt.Fprintln(out, "history is empty")
			return false, err
		}
		for i := 0; i < len(*history); i += 2 {
			if _, err := fmt.Fprintf(out, "  user: %s\n", (*history)[i].text); err != nil {
				return false, err
			}
			if i+1 < len(*history) {
				if _, err := fmt.Fprintf(out, "  assistant: %s\n", (*history)[i+1].text); err != nil {
					return false, err
				}
			}
		}
		return false, nil
	case "/stats":
		// The REPL's whole design claim is that later turns re-attach the previous turn's
		// blocks instead of prefilling them again, and until now nothing in the session let a
		// reader see whether that was happening. On a model whose prompt is shorter than one
		// block it cannot happen at all, and that looked exactly like it working.
		if stats == nil {
			_, err := fmt.Fprintln(out, "stats are not available in this session")
			return false, err
		}
		st := stats()
		if _, err := fmt.Fprintf(out, "  turns:      %d\n", len(*history)/2); err != nil {
			return false, err
		}
		if _, err := fmt.Fprintf(out, "  tokens:     %d generated\n", st.TokensGenerated); err != nil {
			return false, err
		}
		if _, err := fmt.Fprintf(out, "  reused:     %d prompt positions from shared prefixes\n", st.PrefixReused); err != nil {
			return false, err
		}
		return false, nil
	case "/help":
		_, err := fmt.Fprint(out, `Commands:
  /exit, /quit, /q   end the session
  /clear             forget the conversation
  /history           show the conversation so far
  /stats             tokens generated and prefix positions reused
  /help              this message
`)
		return false, err
	default:
		_, err := fmt.Fprintf(out, "unknown command %q, try /help\n", line)
		return false, err
	}
}
