package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
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
	topK := fs.Int("top-k", 40, "top-k sampling cutoff, 0 disables it. Only applies with --temp > 0")
	topP := fs.Float64("top-p", 0, "nucleus sampling cutoff in (0,1], 0 disables it. Only applies with --temp > 0")
	cache := registerCacheFlags(fs, true)
	noPrefix := fs.Bool("no-prefix-cache", false, "disable prefix reuse between turns")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := resolveModel(fs, model); err != nil {
		return err
	}
	if err := cache.validate("chat", true); err != nil {
		return err
	}
	// Sampling is switched on by "temp > 0" and top-k applies at "topK > 0", so a negative
	// value is the same as zero: the session goes greedy without saying so.
	// NaN and Inf pass every comparison below, so each is named rather than left to a range
	// check: a NaN temperature reached the sampler, made sum NaN, failed every `r < acc`
	// and returned the last candidate -- the least likely token in the set.
	if math.IsNaN(*temp) || math.IsInf(*temp, 0) || *temp < 0 {
		return fmt.Errorf("chat: --temp must be a finite, non-negative number (0 means greedy)")
	}
	if *topK < 0 {
		return fmt.Errorf("chat: --top-k must not be negative (0 disables it)")
	}
	if math.IsNaN(*topP) || *topP < 0 || *topP > 1 {
		return fmt.Errorf("chat: --top-p must be between 0 and 1 (0 disables it)")
	}
	// The engine refuses a negative budget too, but only when a turn is submitted -- so the
	// user typed a question, waited for the model to load, and then read an error about the
	// flag. Checking it here reports it before the session starts.
	//
	// Zero is refused as well, which is where chat parts company with run. The engine accepts
	// 0 as "no tokens" and finishes the sequence immediately, so a REPL with --max-tokens 0
	// runs a whole session that answers every question with an empty line: measured against
	// the fixture, a turn printed nothing between the prompt and the next "> ", and /stats
	// then reported "tokens: 0 generated" as though that were a served turn. run's zero is
	// observable -- it is one command that prefills and prints nothing -- while here it is a
	// session that appears broken.
	if *maxTokens < 1 {
		return fmt.Errorf("chat: --max-tokens must be at least 1; a turn that generates " +
			"nothing would answer every question with an empty line")
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
	// Probe the template with a one-turn conversation before choosing the banner.
	//
	// UsesModelTemplate reads renderReason, which is only written when a render fails, so
	// before this probe the banner said "the model's own chat template" for a template that
	// could never render -- and then printed the correction after the first answer, leaving
	// two lines that disagreed. Measured with a template of "{{ messages[99].content }}":
	// banner first, correction after the reply.
	//
	// The probe is a real render rather than a check, because that is what the renderer
	// records: it costs one template evaluation at startup, which a REPL can afford, and it
	// leaves the renderer in the state every later turn will be in anyway.
	renderer.Render([]engine.Turn{{Role: "user", Text: "hi"}})
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
	// The prefix index stores and matches whole blocks, so a block larger than the context can
	// never be filled and the default-on prefix reuse would silently never fire. Same check as
	// serve's and bench's, made here because chat enables reuse by default rather than on
	// request: an oversized --block-size arrives without the operator asking for the feature
	// that breaks.
	//
	// The window comes from llama.LoadConfig, the same source serve and bench use, rather than
	// from cfgForBlocks: that helper answers a different question -- how big to make the pool --
	// and falls back to a constant when the file states no context. A file with no context_length
	// fails inside engine.New anyway, but reading two sources for one concept is how the two
	// checks would drift apart.
	bcfg, err := llama.LoadConfig(gf)
	if err != nil {
		return err
	}
	if blockSize > bcfg.ContextLen {
		return fmt.Errorf("chat: --block-size %d is larger than this model's %d-position "+
			"context, so no block ever fills and prefix reuse could never fire; use "+
			"--block-size %d or less", blockSize, bcfg.ContextLen, bcfg.ContextLen)
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
	if _, err := fmt.Fprintf(out, "Type a message and press enter. Commands: %s\n\n", chatCommands); err != nil {
		return err
	}

	var history []chatTurn
	// notedFallback records that the session has already explained a render-time fallback.
	notedFallback := false
	// notedStuck records which repeating failures the session has already explained. A
	// conversation that no longer fits its window, and one that outgrew the pool, both fail
	// identically on every later turn -- a failed turn does not grow the history -- so the
	// full explanation is worth reading once and is noise after that. Measured on the
	// 128-position fixture: 34 identical overflow paragraphs for 40 inputs; measured with
	// --blocks 4, six identical pool paragraphs for eight.
	//
	// Keyed by the sentinel rather than kept as two booleans, because these are the same
	// defect and the next repeating failure should join them rather than grow a third flag.
	notedStuck := map[error]bool{}
	// notedCut records that the session has already explained a turn the window stopped.
	// Separate from notedFallback because the two are unrelated: a session can hit either
	// without the other, and sharing one flag would silence the second explanation.
	notedCut := false
	var systemTurn string
	if opts.system != "" {
		systemTurn = opts.system
	}

	scanner := bufio.NewScanner(in)
	// One line is one message, so the default 64 KiB line limit is raised: a pasted
	// paragraph often arrives as one very long line.
	//
	// A paste with real newlines is not one message. Each line becomes its own turn, so the
	// history grows by a line at a time and every line resends everything before it. Measured
	// on the 512-position fixture: a twelve-line message -- a request line, ten lines of text,
	// a follow-up question -- produced three turns and then three "conversation is 515 tokens"
	// errors, with the remaining lines dropped. Nothing joins the lines back up, and there is no
	// input mode that would. The line limit is per line, not per conversation, so raising it
	// further would not change this.
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
			if line == "/clear" {
				// A fresh conversation is a fresh explanation: the short form above says
				// "(unchanged; /clear starts over)", so after a /clear the next overflow has to
				// be described in full again or the advice points at a command already used.
				notedStuck = map[error]bool{}
			}
			continue
		}

		turn, err := chatTurnOnce(out, eng, tok, systemTurn, history, line, opts)
		if err != nil {
			// A failed turn leaves the history untouched, so the next question is
			// asked against the same context rather than against a broken one.
			if kind := repeatingFailure(err); kind != nil && notedStuck[kind] {
				// The full text says the next turn fails the same way; once that has been read,
				// repeating it verbatim for every later input is the opposite of help.
				if _, werr := fmt.Fprintf(out, "error: %v (unchanged; /clear starts over)\n", err); werr != nil {
					return werr
				}
			} else {
				if _, werr := fmt.Fprintf(out, "error: %v\n", explainChatError(err)); werr != nil {
					return werr
				}
				if kind != nil {
					notedStuck[kind] = true
				}
			}
			continue
		}
		// The window can be what stopped a turn, and the reply alone does not say so: a turn
		// that asked for 128 tokens and produced 104 reads as a model that finished its
		// sentence, which is what a model stopping on its own looks like too. `run --stats`
		// prints a note for the same case and `bench` reports it in its NOTE column, so this
		// is the one front end that was silent about it. Measured on the 128-position stories
		// model: `chat --max-tokens 128` generated 104 of a requested 128 with nothing said.
		//
		// Announced once per session, like the render fallback below: the limit is a property
		// of the conversation rather than of each turn, and a note under every reply would be
		// noise in exactly the session that has already read it.
		if !notedCut && windowCutTheBudget(turn, opts.maxTokens) {
			notedCut = true
			if _, werr := fmt.Fprintf(out, "(the %d-token window is full, so the %d-token "+
				"budget was cut to %d; /clear or lower --max-tokens)\n",
				turn.contextLen, opts.maxTokens, turn.generated); werr != nil {
				return werr
			}
		}
		history = append(history,
			chatTurn{role: "user", text: line},
			chatTurn{role: "assistant", text: turn.reply},
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

// windowCutTheBudget reports whether the window, rather than the model, ended a turn.
//
// Both ways a turn can end short look identical in the text: a reply that stops because the
// window ran out and one that stops because the model emitted its end-of-sequence token both
// finish mid-sentence or mid-word. What separates them is the engine's finish reason. "length"
// with fewer tokens than the budget asked for means the budget was cut to fit; "stop" with
// fewer means the model chose to stop, and saying the window cut it would be false.
//
// It is a named predicate rather than an inline condition because the second half is the one
// that is easy to get wrong: "generated < budget" alone is true for every reply a model ends
// itself, and a note printed then is worse than no note at all.
func windowCutTheBudget(turn chatTurnResult, budget int) bool {
	return turn.finishReason == "length" && turn.generated < budget
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
// errContextFull marks a turn whose rendered conversation is longer than the model's window.
//
// It is a sentinel so explainChatError can recognise the case: it is the second failure that
// repeats, and the doc there used to call the pool "the one failure here that repeats". The
// window has the same trap for the same reason -- history only grows on a successful turn --
// but arrives as a plain encoding check rather than an engine error, so it needed a marker.
var errContextFull = errors.New("context window full")

// repeatingFailure classifies the failures that repeat identically until the conversation is
// cleared, returning the sentinel that identifies the kind or nil for a one-off error.
//
// Both kinds share the property that matters here: the next turn is asked against the same
// history, so it fails in the same way, and the full advice is worth reading exactly once.
// errors.Is is the test rather than equality, because each comes wrapped in the context that
// names the sizes.
func repeatingFailure(err error) error {
	switch {
	case errors.Is(err, errContextFull):
		return errContextFull
	case errors.Is(err, llama.ErrPoolExhausted):
		return llama.ErrPoolExhausted
	default:
		return nil
	}
}

// explainChatError adds what the caller can do about an error the engine reports bare.
//
// Two failures repeat, for the same reason: the history is not advanced on a failed turn, so
// the next turn submits the same conversation against the same limit and fails identically for
// as long as the session lasts. Exhausting the pool is one. The other is the conversation
// outgrowing the model's window, which was missed here because it is an encoding check rather
// than an engine error: measured on a 256-position fixture, three turns filled the window and
// each of the next nine attempts printed the same "conversation is 318 tokens, context is 256"
// with nothing to say that the situation could not change on its own. The engine's message names the pool's size, which
// is the wrong end of the problem -- the session already knows what it asked for -- and suggests
// nothing.
//
// The pool is sized for one full context by default, so this only happens when --blocks was set
// below that.
//
// /clear is offered as a way out only when it can be one. It empties the history, so the next
// prompt is just the new message -- but the message the user already typed is part of the prompt
// too, and if that alone exceeds the pool then /clear changes nothing and the same error comes
// back. The previous wording called /clear "the way out" unconditionally, which was wrong for
// that case: measured on a 60-word message against --blocks 4, the error repeated verbatim after
// /clear. Saying when it helps costs one clause and stops the advice from being a dead end.
func explainChatError(err error) error {
	// The window can be outgrown by the conversation itself, which is a different failure
	// from the pool running out but has the same trap: a failed turn does not grow the
	// history, so the next one fails identically. Without saying so the REPL answers every
	// later message with the same line -- measured on a 256-position fixture, three turns
	// filled the window and the next nine attempts each printed "conversation is 318 tokens,
	// context is 256" with no hint that nothing had changed.
	if errors.Is(err, errContextFull) {
		return fmt.Errorf("%w -- this conversation no longer fits the model's window, and the"+
			" next turn will fail the same way because a failed turn does not grow the history;"+
			" /clear starts over, and the model's window is fixed, so a longer --system or a"+
			" smaller --max-tokens only helps by leaving room for fewer turns", err)
	}
	if !errors.Is(err, llama.ErrPoolExhausted) {
		return err
	}
	// The engine's message already gives the pool's size, so this adds only what it cannot know:
	// that the session repeats the failure, and what actually helps.
	return fmt.Errorf("%w -- this conversation outgrew the pool, and the next turn will fail the"+
		" same way because a failed turn does not grow the history; /clear forgets the history,"+
		" which helps if the history is what overflowed but not if the message alone is longer"+
		" than the pool; raising --blocks, or leaving it unset to size the pool for the whole"+
		" context, always helps", err)
}

// chatTurnResult is what one turn produced beyond its text.
//
// The text is the reply; the two figures beside it describe how the turn ended. They exist
// because a turn can stop for a reason the text cannot show: finishing on an end-of-sequence
// token and being cut off by the window both end mid-sentence or mid-word, and only
// finishReason plus the count of what was generated tells them apart.
type chatTurnResult struct {
	// reply is the assistant's text, exactly what was printed.
	reply string
	// finishReason is the engine's reason for stopping: "stop" (an end-of-sequence token
	// or a stop id), "length" (the budget or the window ran out), "cancel", "abandoned",
	// or "error".
	finishReason string
	// generated is how many tokens the sequence produced in total, from the terminal
	// event. It is the engine's figure rather than the number of events this loop read,
	// for the reason Event.Generated documents.
	generated int
	// contextLen is the model's window, which is what a "length" finish ran into.
	contextLen int
}

func chatTurnOnce(out io.Writer, eng *engine.Engine, tok *tokenize.Tokenizer, system string, history []chatTurn, user string, opts chatOptions) (chatTurnResult, error) {
	prompt := opts.renderer.Render(chatTurns(system, history, user))
	cfg := eng.Model().Config()
	// The chat transcript is markup, not prose: its markers are control tokens the
	// model was trained to see, so they have to be matched whole.
	ids := tok.EncodeWithSpecials(prompt, cfg.BOSID >= 0)
	if len(ids) == 0 {
		return chatTurnResult{}, fmt.Errorf("prompt encoded to no tokens")
	}
	if len(ids) >= cfg.ContextLen {
		return chatTurnResult{}, fmt.Errorf("%w: conversation is %d tokens, context is %d",
			errContextFull, len(ids), cfg.ContextLen)
	}

	sampler := llama.Sampler(llama.Greedy{})
	if opts.temp > 0 {
		// math/rand's generator, not crypto/rand: sampling quality does not depend on
		// unpredictability, and a REPL benefits from the cheap path.
		//
		// There is no --seed here, unlike `orbix run`, and it is a decision rather than an
		// omission: a session's prompt depends on every previous turn, so reproducing one
		// would mean replaying the whole conversation against the same generator draws. A
		// single-run seed gives the impression of reproducibility it cannot deliver. `run`
		// has one because there the prompt is fixed and the seed is the only variable.
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
		return chatTurnResult{}, err
	}

	// The reply accumulates in the result rather than in a second builder: the history has to
	// keep exactly the text that was printed, and two builders made that agreement a thing to
	// maintain rather than a thing that holds by construction.
	var ended chatTurnResult
	dec := tok.Seq()
	// The engine advances sequences only inside Pump, so this loop drives the model
	// and drains the channel in one goroutine. Nothing else can make progress, which
	// is why a REPL needs no locks and no goroutine per token.
	//
	// Each round: emit whatever the channel already holds, then let the engine run
	// one scheduling round. Pump reports how many sequences are still active, so a
	// zero means the sequence finished this round and the remainder can be drained.
	for {
		if err := drainChatEvents(ch, out, dec, &ended); err != nil {
			return chatTurnResult{}, err
		}
		if eng.Pump() == 0 {
			// The final round may have emitted tokens before the sequence closed; the
			// channel is closed by now, so this drains to completion and returns.
			for ev := range ch {
				if ev.Err != nil {
					return chatTurnResult{}, ev.Err
				}
				if ev.HasToken {
					text := dec.Token(ev.Token)
					if _, err := fmt.Fprint(out, text); err != nil {
						return chatTurnResult{}, err
					}
					ended.reply += text
				}
				if ev.Done {
					ended.finishReason = ev.FinishReason
					ended.generated = ev.Generated
				}
			}
			// A character split across the last tokens is still held by the decoder,
			// waiting for a byte that never arrives. Dropping it would make the printed
			// reply differ from the non-streaming decode of the same tokens.
			if tail := dec.Flush(); tail != "" {
				if _, err := fmt.Fprint(out, tail); err != nil {
					return chatTurnResult{}, err
				}
				ended.reply += tail
			}
			if _, err := fmt.Fprintln(out); err != nil {
				return chatTurnResult{}, err
			}
			ended.contextLen = cfg.ContextLen
			return ended, nil
		}
	}
}

// drainChatEvents prints every event already queued, returning the sequence's error if
// it reported one.
//
// It must not block: more events only arrive while Pump runs, so waiting here would
// deadlock against the engine rather than make progress.
func drainChatEvents(ch <-chan engine.Event, out io.Writer, dec *tokenize.StreamDecoder, ended *chatTurnResult) error {
	for {
		select {
		case ev, ok := <-ch:
			if !ok {
				// The channel closed, so no more tokens can complete a held-back character.
				if tail := dec.Flush(); tail != "" {
					if _, err := fmt.Fprint(out, tail); err != nil {
						return err
					}
					ended.reply += tail
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
				ended.reply += text
			}
			// The terminal event can arrive in a round that still had tokens to drain, so
			// it is recorded here as well as in the closing loop -- otherwise a turn whose
			// stop landed one round early would report no reason at all.
			if ev.Done {
				ended.finishReason = ev.FinishReason
				ended.generated = ev.Generated
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
// chatHelp is the command list, and it is the only copy: the startup banner prints the
// same names on one line rather than restating them.
//
// Two lists existed and drifted -- /stats was added for the prefix-reuse claim and the banner
// never learned about it, so the one command that shows whether reuse is happening was
// reachable only from a command the banner did not mention.
const chatHelp = `Commands:
  /exit, /quit, /q   end the session
  /clear             forget the conversation
  /history           show the conversation so far
  /stats             tokens, prefix positions reused, and KV pool use
  /help              this message
`

// chatCommands is the one-line form of chatHelp for the startup banner.
//
// A const cannot be derived from chatHelp at compile time, so the two are kept in step by a
// test that checks every name in this line appears in chatHelp (and that this line does not
// claim one chatHelp omits). Restating the names is unavoidable; leaving the restatement
// unchecked is not.
const chatCommands = "/exit, /clear, /history, /stats, /help"

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
		// turns is the live conversation; the two counters below it are totals for the whole
		// session. Printing them side by side without saying so made /clear look broken: it
		// reset turns to 0 while tokens and reused stayed put, which reads as stale numbers
		// rather than as a different question being answered. The labels say which is which.
		if _, err := fmt.Fprintf(out, "  turns:      %d in this conversation\n",
			len(*history)/2); err != nil {
			return false, err
		}
		if _, err := fmt.Fprintf(out, "  tokens:     %d generated this session\n",
			st.TokensGenerated); err != nil {
			return false, err
		}
		if _, err := fmt.Fprintf(out, "  reused:     %d prompt positions from shared prefixes this session\n",
			st.PrefixReused); err != nil {
			return false, err
		}
		// The pool, when the session has one, because running out of it is the one failure
		// this REPL hits repeatedly: a conversation outgrows the pool, every later turn fails
		// the same way because a failed turn does not grow the history, and the error message
		// names the size but not how close to it the session is.
		//
		// The total is printed alongside the free count because the two answer different
		// questions, and the free count alone was misleading: a session whose every turn is
		// refused shows "4 of 4 blocks free", and a reader concludes the pool is fine. It is
		// fine -- the request is simply larger than the pool can ever hold, which is the case
		// explainChatError already says /clear cannot fix. Printing the size is what lets them
		// tell that apart from contention.
		//
		// Only when a pool exists: a contiguous cache has no blocks, and printing
		// "blocks:     0 / 0" on every session would be noise in exactly the configuration
		// that cannot hit this failure.
		if st.PoolBlocks > 0 {
			if _, err := fmt.Fprintf(out, "  kv pool:    %d of %d blocks free (%d positions each, %d total)\n",
				st.PoolFree, st.PoolBlocks, st.PoolBlockSize,
				st.PoolBlocks*st.PoolBlockSize); err != nil {
				return false, err
			}
		}
		return false, nil
	case "/help":
		_, err := fmt.Fprint(out, chatHelp)
		return false, err
	default:
		_, err := fmt.Fprintf(out, "unknown command %q, try /help\n", line)
		return false, err
	}
}
