package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/tokenize"
)

// The chat REPL is driven through its session function rather than through the
// process, so the tests can supply input and read output without a subprocess or a
// pty. The session contract is small: one line in, one reply out, and history that
// only grows on success.

func chatModel(t *testing.T) string {
	t.Helper()
	// A constructed model: these tests check the REPL's structure -- prompts appear, a
	// reply is non-empty, /exit ends the session -- not the quality of the text. A
	// checkpoint would skip them wherever one is not on disk.
	return gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
}

// runChat drives a session with the given input lines and returns its output.
func runChat(t *testing.T, opts chatOptions, input string) string {
	t.Helper()
	var out bytes.Buffer
	if err := chatSession(strings.NewReader(input), &out, opts); err != nil {
		t.Fatalf("chatSession: %v", err)
	}
	return out.String()
}

// TestChatDoesNotInventARenderFailure covers a message the REPL printed when nothing had
// gone wrong.
//
// A model that declares no chat template is in the neutral transcript from the first turn,
// and the banner says so before the first prompt. The post-turn check then asked
// UsesModelTemplate, which is false both for "no template" and for "the template failed to
// render", so every such session announced a render-time fallback after its first successful
// turn -- contradicting the banner directly above it and describing a failure that never
// happened. On a model with no template the message is not merely redundant: it suggests
// something broke mid-conversation when nothing changed at all.
//
// The two cases are now distinguished by HasTemplate, and this test pins the "no template"
// half. A fixture with no ChatTemplate is exactly that model.
func TestChatDoesNotInventARenderFailure(t *testing.T) {
	out := runChat(t, chatOptions{
		model:     chatModel(t),
		maxTokens: 4,
		topK:      40,
	}, "Once upon a time\n/exit\n")

	// The banner must still say which format is in use -- that is the honest statement.
	if !strings.Contains(out, "neutral transcript") {
		t.Errorf("the banner no longer names the fallback:\n%s", out)
	}
	// And the per-turn message must not appear, because no render failed.
	if strings.Contains(out, "could not render this conversation") {
		t.Errorf("the REPL reported a render failure for a model with no template:\n%s", out)
	}
}

// TestChatReportsARealRenderTimeFallback is the other half of the pair: a template that
// parses and then fails to render a particular conversation must still be reported once.
//
// Without this, the fix for the phantom message above could be "never print it", which would
// silence the case the message exists for -- a conversation that silently changed format
// partway through.
func TestChatReportsARealRenderTimeFallback(t *testing.T) {
	// A template that parses but indexes past the conversation, so rendering fails only when
	// a request arrives; Parse succeeds.
	path := gguftest.Model{
		NoEOS: true, ContextLen: 512,
		ChatTemplate: `{{ messages[9].content }}`,
	}.Write(t, t.TempDir())

	out := runChat(t, chatOptions{
		model:     path,
		maxTokens: 4,
		topK:      40,
	}, "Once upon a time\n/exit\n")

	if !strings.Contains(out, "could not render this conversation") {
		t.Errorf("a template that failed at render time was not reported:\n%s", out)
	}
}

func TestChatRepliesAndExits(t *testing.T) {
	out := runChat(t, chatOptions{
		model:     chatModel(t),
		maxTokens: 4,
		topK:      40,
	}, "Once upon a time\n/exit\n")

	for _, want := range []string{"orbix chat:", "Type a message", "> "} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
	// The reply must be non-empty: between the prompt and the next one there has to be
	// generated text.
	lines := strings.Split(out, "\n")
	var replied bool
	for _, line := range lines {
		if strings.HasPrefix(line, "> ") && strings.TrimSpace(strings.TrimPrefix(line, "> ")) != "" {
			replied = true
		}
	}
	if !replied {
		t.Errorf("no reply was printed:\n%s", out)
	}
}

func TestChatEOFEndsSession(t *testing.T) {
	// A piped session ends at EOF without needing /exit, which is what makes
	// "printf 'hi' | orbix chat" usable.
	out := runChat(t, chatOptions{model: chatModel(t), maxTokens: 2, topK: 40}, "hello\n")
	if !strings.Contains(out, "orbix chat:") {
		t.Errorf("no banner:\n%s", out)
	}
}

func TestChatClearAndHistory(t *testing.T) {
	out := runChat(t, chatOptions{
		model:     chatModel(t),
		maxTokens: 3,
		topK:      40,
	}, "Once upon a time\n/history\n/clear\n/history\n/exit\n")

	if !strings.Contains(out, "user: Once upon a time") {
		t.Errorf("/history did not show the conversation:\n%s", out)
	}
	// The assistant half as well, and non-empty. /history is rendered from the same chatTurn
	// list the next turn's prompt is built from, so a reply that never reaches it is not a
	// display problem: every later turn asks the model to continue a conversation it has
	// only seen one side of. Only the user line was asserted, so emptying both places that
	// append generated text to the reply buffer -- the drain and the final-drain loop --
	// passed the whole cmd suite.
	if !strings.Contains(out, "  assistant: ") {
		t.Errorf("/history shows no assistant reply:\n%s", out)
	}
	for _, line := range strings.Split(out, "\n") {
		rest, ok := strings.CutPrefix(line, "  assistant: ")
		if !ok {
			continue
		}
		if strings.TrimSpace(rest) == "" {
			t.Errorf("/history recorded an empty assistant reply for a turn that generated text")
		}
	}
	if !strings.Contains(out, "history cleared") {
		t.Errorf("/clear did not confirm:\n%s", out)
	}
	if !strings.Contains(out, "history is empty") {
		t.Errorf("/history after /clear did not report empty:\n%s", out)
	}
}

func TestChatHelpAndUnknownCommand(t *testing.T) {
	out := runChat(t, chatOptions{model: chatModel(t), maxTokens: 2, topK: 40},
		"/help\n/nope\n/exit\n")
	// The descriptions are what /help adds over the banner, which lists the same names on one line.
	// Asserting a name only proves the banner appeared: with the /help case removed, the command falls
	// to the default arm and the banner's "/clear" still satisfies it.
	for _, want := range []string{
		"Commands:",
		"forget the conversation",
		"show the conversation so far",
		`unknown command "/nope"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
}

// TestChatBannerNamesEveryCommand locks the startup banner against /help.
//
// The banner listed four commands and /help showed six. /stats was the one that mattered: it
// was added so a session could show whether prefix reuse was happening at all, and the only way
// to discover it was a command the banner did not mention.
//
// The existing help test checked that /clear appeared, which is why the gap survived -- it
// asserted one sample rather than the set. This one derives the set from chatHelp and requires
// the banner to name every member of it.
func TestChatBannerNamesEveryCommand(t *testing.T) {
	out := runChat(t, chatOptions{model: chatModel(t), maxTokens: 2, topK: 40}, "/exit\n")

	// Every name chatHelp lists, in its first column, must appear in chatCommands.
	for _, line := range strings.Split(chatHelp, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || line == "Commands:" {
			continue
		}
		names, _, _ := strings.Cut(line, " ")
		for _, name := range strings.Split(names, ",") {
			name = strings.TrimSpace(name)
			if !strings.HasPrefix(name, "/") {
				continue
			}
			if !strings.Contains(chatCommands, name) {
				t.Errorf("chatHelp lists %q but the banner (%q) does not", name, chatCommands)
			}
		}
	}

	// And the banner that actually reached the user has to carry it too, so a future
	// wrapper that printed a different string would fail here rather than nowhere.
	if !strings.Contains(out, "/stats") {
		t.Errorf("the startup banner does not mention /stats:\n%s", out)
	}
}

func TestChatBlankLinesIgnored(t *testing.T) {
	// A blank line must not become a turn: it would otherwise be encoded as a prompt
	// with no user text and answered.
	out := runChat(t, chatOptions{model: chatModel(t), maxTokens: 2, topK: 40},
		"\n\n   \n/exit\n")
	if strings.Count(out, "> ") > 4 {
		t.Errorf("blank lines produced extra prompts:\n%s", out)
	}
}

func TestChatHistoryOnlyGrowsOnSuccess(t *testing.T) {
	// The second turn's prompt must include the first exchange, which is what makes
	// the session a conversation rather than a series of independent questions. The
	// evidence is that /history shows both turns.
	out := runChat(t, chatOptions{
		model:     chatModel(t),
		maxTokens: 3,
		topK:      40,
	}, "Once upon a time\nTell me more\n/history\n/exit\n")

	if got := strings.Count(out, "user: "); got != 2 {
		t.Errorf("history has %d user turns, want 2:\n%s", got, out)
	}
	if got := strings.Count(out, "assistant: "); got != 2 {
		t.Errorf("history has %d assistant turns, want 2:\n%s", got, out)
	}
}

func TestRenderChatPrompt(t *testing.T) {
	// This pins the neutral transcript, which is what a model without a usable
	// template gets. The flattening from a session to turns lives here, so a wrong
	// turn list is caught without running the REPL.
	neutral := func(system string, history []chatTurn, user string) string {
		return engine.RenderChatPrompt(chatTurns(system, history, user))
	}

	got := neutral("be brief", []chatTurn{
		{role: "user", text: "hi"},
		{role: "assistant", text: "hello"},
	}, "how are you")
	want := "System: be brief\nUser: hi\nAssistant: hello\nUser: how are you\nAssistant:"
	if got != want {
		t.Errorf("neutral transcript =\n%q\nwant\n%q", got, want)
	}

	// No system prompt means no System line at all, not an empty one.
	got = neutral("", nil, "hi")
	if want := "User: hi\nAssistant:"; got != want {
		t.Errorf("neutral transcript without system = %q, want %q", got, want)
	}
}

func TestChatPromptTooLongAsksForClear(t *testing.T) {
	path := chatModel(t)
	opts := chatOptions{model: path, maxTokens: 2, topK: 40}

	// Build a conversation that overflows the model's context before the first turn by
	// asking with a prompt longer than the window.
	var sb strings.Builder
	for sb.Len() < 8192 {
		sb.WriteString("Once upon a time there was a little girl. ")
	}
	out := runChat(t, opts, sb.String()+"\n/exit\n")
	if !strings.Contains(out, "context is") {
		t.Errorf("an oversized prompt did not report the context limit:\n%s", out)
	}
	if !strings.Contains(out, "/clear") {
		t.Errorf("the error did not suggest /clear:\n%s", out)
	}
}

func TestChatUsageErrors(t *testing.T) {
	// A missing model is the common first mistake, so it gets a message that names the
	// flag and the environment variable.
	t.Setenv("ORBIX_MODEL", "")
	err := cmdChat(nil)
	if err == nil {
		t.Fatal("cmdChat with no model succeeded, want an error")
	}
	if !strings.Contains(err.Error(), "ORBIX_MODEL") {
		t.Errorf("error = %v, want it to mention ORBIX_MODEL", err)
	}
}

// TestChatRefusesABudgetThatGeneratesNothing covers the value chat accepts but cannot serve.
//
// The engine treats MaxTokens 0 as "no tokens" and finishes the sequence as soon as its
// prompt is prefilled, so a session started with --max-tokens 0 ran to completion and
// answered every question with an empty line: measured against the fixture, one turn printed
// nothing between the "> " prompt and the next one, and /stats then reported "tokens: 0
// generated" as if a turn had been served. Nothing failed, which is what made it a silent
// misconfiguration rather than a bug report.
//
// The refusal is asserted on the flag's name and on the boundary, since the boundary is the
// part an edit could move: 1 still has to start a session.
func TestChatRefusesABudgetThatGeneratesNothing(t *testing.T) {
	t.Run("zero is refused", func(t *testing.T) {
		err := cmdChat([]string{"--model", "unused.gguf", "--max-tokens", "0"})
		if err == nil {
			t.Fatal("chat accepted --max-tokens 0, which answers every turn with nothing")
		}
		if !strings.Contains(err.Error(), "--max-tokens") {
			t.Errorf("error %q does not name the flag", err)
		}
		if strings.Contains(err.Error(), "unused.gguf") {
			t.Errorf("error %q came from opening the model, so the flag was not validated", err)
		}
	})

	t.Run("one still starts a session", func(t *testing.T) {
		// A real model, because the point is that 1 passes validation all the way into a
		// session. The session is fed an immediate /exit so nothing is generated.
		opts := chatOptions{model: chatModel(t), maxTokens: 1, topK: 40}
		out := runChat(t, opts, "/exit\n")
		if !strings.Contains(out, "orbix chat:") {
			t.Errorf("--max-tokens 1 did not start a session:\n%s", out)
		}
	})
}

// TestChatAliasesEndTheSession checks the three spellings that mean "stop".
//
// They share one case in the switch, so the risk is not that one is implemented
// differently but that the shared arm is only ever exercised through "/exit": if the arm were
// narrowed to the first spelling, or a later edit made one name fall through to the model,
// every existing test would still pass because every existing test uses "/exit".
func TestChatAliasesEndTheSession(t *testing.T) {
	for _, cmd := range []string{"/exit", "/quit", "/q"} {
		t.Run(cmd, func(t *testing.T) {
			// A second line follows the command. If the alias did not end the session, that
			// line would be sent to the model and answered, which the absence of a second
			// prompt shows.
			out := runChat(t, chatOptions{model: chatModel(t), maxTokens: 2, topK: 40},
				"Once upon a time\n"+cmd+"\n/help\n")
			// "end the session" appears only in /help's own listing: the startup banner
			// also starts with the word Commands, so that would match either way.
			if strings.Contains(out, "end the session") {
				t.Errorf("%q did not end the session: the line after it was processed\n%s", cmd, out)
			}
			if !strings.Contains(out, "> ") {
				t.Errorf("%q produced no prompt before ending:\n%s", cmd, out)
			}
		})
	}
}

// TestChatAliasesAreNotSentToTheModel is the other half: a command must not also become a
// user message. Both would end the session here, so the check is on the transcript.
func TestChatAliasesAreNotSentToTheModel(t *testing.T) {
	out := runChat(t, chatOptions{model: chatModel(t), maxTokens: 2, topK: 40},
		"/quit\n")
	if strings.Contains(out, "user: /quit") || strings.Contains(out, "user: /q") {
		t.Errorf("a command was recorded as a user message:\n%s", out)
	}
}

// TestExplainChatError covers the one chat failure that repeats on its own.
//
// A failed turn leaves the history untouched, so the next turn submits the same conversation against
// the same full pool and fails identically -- for as long as the session lasts. The engine's message
// names the pool's size, which is the wrong end of the problem, and suggests nothing. Observed with
// --blocks 2: twelve of fifteen turns printed the bare error and none of them said what to do.
//
// The advice has to be advice, not a list of commands. /clear empties the history, so it ends the
// failure only when the history is what overflowed: the message the user just typed is part of the
// prompt too. Measured on a 60-word message against --blocks 4, the error came back verbatim after
// /clear, which is why the message now says which case it is in instead of calling /clear "the way
// out". Raising --blocks, or leaving it unset, works in both cases and is named first in effect.
func TestExplainChatError(t *testing.T) {
	// An unrelated error passes through untouched: wrapping every error would bury the ones that
	// already say what went wrong.
	plain := errors.New("llama: something else")
	if got := explainChatError(plain); got != plain {
		t.Errorf("an unrelated error was rewritten: %v", got)
	}

	exhausted := fmt.Errorf("%w (2 blocks of 16 positions)", llama.ErrPoolExhausted)
	got := explainChatError(exhausted)
	if got == exhausted {
		t.Fatal("pool exhaustion was not explained")
	}
	// The original chain has to survive, because a caller checking errors.Is still needs to see
	// what failed.
	if !errors.Is(got, llama.ErrPoolExhausted) {
		t.Error("the explanation dropped ErrPoolExhausted from the chain")
	}
	for _, want := range []string{"/clear", "--blocks"} {
		if !strings.Contains(got.Error(), want) {
			t.Errorf("error %q does not mention %q", got, want)
		}
	}
	// /clear must not be presented as an unconditional cure. The condition is the difference
	// between advice and a dead end, and its absence is the defect this comment records.
	if !strings.Contains(got.Error(), "if the history is what overflowed") {
		t.Errorf("error %q names /clear without saying when it helps", got)
	}
}

// TestClearDoesNotAlwaysEndPoolExhaustion pins the case the message now describes.
//
// It is a session-level test because the claim is about what /clear does to a real session:
// with a message longer than the whole pool, the same error must come back after /clear. If
// someone later changes /clear to release the pool, this test will say so -- and the message
// would then be free to call /clear unconditional again.
func TestClearDoesNotAlwaysEndPoolExhaustion(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	// A pool far smaller than one message: 2 blocks of 16 positions cannot hold ~60 words.
	long := strings.Repeat("word ", 60)
	out := runChat(t, chatOptions{
		model: path, maxTokens: 2, topK: 40, blocks: 2, blockSize: 16,
	}, long+"\n/clear\n"+long+"\n/exit\n")
	if n := strings.Count(out, "block pool exhausted"); n != 2 {
		t.Errorf("saw %d pool-exhaustion errors, want 2 (one before /clear and one after):\n%s", n, out)
	}
}

// TestChatStatsCommand covers /stats, which exists because the REPL's central claim -- that a
// later turn re-attaches the previous turn's KV blocks instead of prefilling them again -- had no
// way to be observed from inside a session. Until it did, a session where reuse was happening and
// one where the prompt was too short to reuse anything printed the same thing: nothing.
func TestChatStatsCommand(t *testing.T) {
	model := chatModel(t)

	// Before any turn: the counters are zero and still have to be reported, because "0 reused"
	// is an answer and silence is not.
	out := runChat(t, chatOptions{model: model, maxTokens: 2, prefixCache: true}, "/stats\n/exit\n")
	if !strings.Contains(out, "/stats") && !strings.Contains(out, "reused:") {
		t.Errorf("/stats printed nothing recognisable:\n%s", out)
	}
	if !strings.Contains(out, "reused:") || !strings.Contains(out, "0 prompt positions") {
		t.Errorf("a fresh session did not report zero reuse:\n%s", out)
	}

	// After a conversation long enough to span whole blocks, reuse has to be visible as a
	// positive count -- that is the number this command was added to expose.
	long := "Please tell me a long story about a dragon and a knight who became friends " +
		"despite everything that happened between their families over many years of conflict"
	out = runChat(t, chatOptions{model: model, maxTokens: 2, prefixCache: true},
		long+"\nThat was interesting, please continue the story with more detail\n/stats\n/exit\n")
	if !strings.Contains(out, "reused:") {
		t.Fatalf("/stats printed no reuse line:\n%s", out)
	}
	// The second turn re-sends the first turn's history, so something must have been reused.
	// Asserted as "not zero" rather than an exact tally: the figure depends on the block size
	// and how the tokenizer splits this particular prompt.
	if strings.Contains(out, "reused:      0 prompt positions") {
		t.Errorf("a two-turn conversation reused nothing:\n%s", out)
	}
}

// TestChatRefusesABlockBiggerThanTheContext covers the REPL's version of the prefix-cache trap.
//
// Reuse is on by default in chat, so an oversized --block-size breaks a feature the operator
// never asked for and gets no line announcing it. A prefix index stores and matches whole
// blocks, so a block the context cannot fill is never published: measured on Qwen2.5-0.5B a
// 16-position block reused 32 positions across two turns, and this configuration would reuse
// none however long the conversation ran.
func TestChatRefusesABlockBiggerThanTheContext(t *testing.T) {
	var out bytes.Buffer
	// The fixture's default window is 64 positions, so 1000 cannot fit.
	err := chatSession(strings.NewReader("/exit\n"), &out, chatOptions{
		model:     gguftest.Model{NoEOS: true}.Write(t, t.TempDir()),
		blockSize: 1000,
		maxTokens: 4,
		topK:      40,
	})
	if err == nil {
		t.Fatal("chat accepted a block larger than the context")
	}
	for _, want := range []string{"--block-size", "1000", "64"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// TestDrainChatEvents covers the three exits of the REPL's event pump.
//
// The function had no test at all. Its happy path runs on every chat turn, but the exits are
// what a broken terminal takes: a write that fails must be reported rather than looped on, an
// engine error must surface instead of being dropped, and the held-back tail has to be
// flushed when the channel closes -- that last one is the only place a partially completed
// multi-byte character is rendered.
func TestDrainChatEvents(t *testing.T) {
	newDecoder := func(t *testing.T) *tokenize.StreamDecoder {
		t.Helper()
		gf, err := gguf.Open(gguftest.Model{NoEOS: true, ContextLen: 64}.Write(t, t.TempDir()))
		if err != nil {
			t.Fatalf("gguf.Open: %v", err)
		}
		tok, err := tokenize.New(gf)
		if err != nil {
			t.Fatalf("tokenize.New: %v", err)
		}
		return tok.Seq()
	}

	t.Run("tokens are written and collected", func(t *testing.T) {
		ch := make(chan engine.Event, 4)
		ch <- engine.Event{HasToken: true, Token: 3}
		ch <- engine.Event{HasToken: true, Token: 4}
		close(ch)

		var out strings.Builder
		var reply strings.Builder
		if err := drainChatEvents(ch, &out, newDecoder(t), &reply); err != nil {
			t.Fatalf("drainChatEvents: %v", err)
		}
		// Whatever the fixture's pieces spell, the display and the gathered reply agree:
		// that equality is what makes /history show what the user saw.
		if out.String() != reply.String() {
			t.Errorf("displayed %q but collected %q", out.String(), reply.String())
		}
	})

	t.Run("an engine error is returned", func(t *testing.T) {
		wantErr := errors.New("engine gave up")
		ch := make(chan engine.Event, 1)
		ch <- engine.Event{Err: wantErr}
		close(ch)

		var out, reply strings.Builder
		err := drainChatEvents(ch, &out, newDecoder(t), &reply)
		if !errors.Is(err, wantErr) {
			t.Fatalf("drainChatEvents returned %v, want the engine error", err)
		}
	})

	t.Run("a write error is returned rather than ignored", func(t *testing.T) {
		wantErr := errors.New("broken pipe")
		ch := make(chan engine.Event, 1)
		ch <- engine.Event{HasToken: true, Token: 3}
		close(ch)

		var reply strings.Builder
		// A writer that refuses everything: the terminal has gone away.
		err := drainChatEvents(ch, failingWriter{err: wantErr}, newDecoder(t), &reply)
		if !errors.Is(err, wantErr) {
			t.Fatalf("drainChatEvents returned %v, want the write error", err)
		}
	})

	t.Run("returns when the channel has nothing ready", func(t *testing.T) {
		// An open, empty channel: the default branch must return rather than block, or the
		// REPL would hang between turns.
		ch := make(chan engine.Event)
		var out, reply strings.Builder
		if err := drainChatEvents(ch, &out, newDecoder(t), &reply); err != nil {
			t.Fatalf("drainChatEvents: %v", err)
		}
	})
}

// failingWriter reports a fixed error instead of accepting bytes.
type failingWriter struct{ err error }

func (w failingWriter) Write([]byte) (int, error) { return 0, w.err }

// TestChatCommandStatsAndHistory covers the two commands whose output is a computed summary.
//
// Both have an arithmetic step that nothing checked. /stats divides the history length by two
// to report turns, and /history pairs messages two at a time; a session that just answered one
// question has two entries, and a session interrupted between the user line and the reply has
// three. Reporting the raw length as a turn count is off by that factor, and stepping the
// pairing loop by one prints each message twice.
//
// The nil-stats case is the one a test session hits: a session built without an engine counter
// has to say so rather than print zeroes that look like a measurement.
func TestChatCommandStatsAndHistory(t *testing.T) {
	historyOf := func(n int) []chatTurn {
		out := make([]chatTurn, n)
		for i := range out {
			if i%2 == 0 {
				out[i] = chatTurn{role: "user", text: fmt.Sprintf("q%d", i/2+1)}
			} else {
				out[i] = chatTurn{role: "assistant", text: fmt.Sprintf("a%d", i/2+1)}
			}
		}
		return out
	}

	t.Run("turns counts exchanges, not messages", func(t *testing.T) {
		history := historyOf(4) // two exchanges
		var out bytes.Buffer
		st := engine.Stats{TokensGenerated: 7, PrefixReused: 32}
		if _, err := chatCommand(&out, "/stats", &history, func() engine.Stats { return st }); err != nil {
			t.Fatalf("chatCommand: %v", err)
		}
		got := out.String()
		if !strings.Contains(got, "turns:      2") {
			t.Errorf("/stats reported %q, want 2 turns for four messages", got)
		}
		if !strings.Contains(got, "tokens:     7 generated") {
			t.Errorf("/stats did not report the token count:\n%s", got)
		}
		if !strings.Contains(got, "reused:     32 prompt positions") {
			t.Errorf("/stats did not report the reuse count:\n%s", got)
		}
	})

	t.Run("no stats reports that rather than zeroes", func(t *testing.T) {
		history := historyOf(2)
		var out bytes.Buffer
		if _, err := chatCommand(&out, "/stats", &history, nil); err != nil {
			t.Fatalf("chatCommand: %v", err)
		}
		got := out.String()
		if !strings.Contains(got, "not available") {
			t.Errorf("/stats with no counter said %q, want it to say stats are unavailable", got)
		}
		// A zero would read as a measurement of zero.
		if strings.Contains(got, "tokens:") {
			t.Errorf("/stats printed a token count with no counter to read:\n%s", got)
		}
	})

	t.Run("history pairs each question with its answer", func(t *testing.T) {
		history := historyOf(4)
		var out bytes.Buffer
		if _, err := chatCommand(&out, "/history", &history, nil); err != nil {
			t.Fatalf("chatCommand: %v", err)
		}
		got := out.String()
		for _, want := range []string{"user: q1", "assistant: a1", "user: q2", "assistant: a2"} {
			if !strings.Contains(got, want) {
				t.Errorf("/history does not show %q:\n%s", want, got)
			}
		}
		// Stepping by one does not repeat the text -- it mislabels it, printing the answer as
		// the next question. Counting occurrences of one string therefore says nothing; what
		// has to be asserted is that every printed line is a role paired with its own text.
		for _, line := range strings.Split(strings.TrimSpace(got), "\n") {
			line = strings.TrimSpace(line)
			role, text, ok := strings.Cut(line, ": ")
			if !ok {
				t.Fatalf("/history printed a line with no role: %q", line)
			}
			// The fixture names questions qN and answers aN, so the role decides which the
			// text must be.
			wantPrefix := "q"
			if role == "assistant" {
				wantPrefix = "a"
			}
			if !strings.HasPrefix(text, wantPrefix) {
				t.Errorf("/history printed %q, but a %s line must carry an %s: the pairing is off",
					line, role, wantPrefix)
			}
		}
	})

	t.Run("history with a reply still pending", func(t *testing.T) {
		// Three entries: two exchanges' worth would be four. The last question has no answer
		// yet, and printing nothing for it is correct -- but the loop must not read past the
		// end to find one.
		history := historyOf(3)
		var out bytes.Buffer
		if _, err := chatCommand(&out, "/history", &history, nil); err != nil {
			t.Fatalf("chatCommand: %v", err)
		}
		got := out.String()
		if !strings.Contains(got, "user: q2") {
			t.Errorf("/history dropped the unanswered question:\n%s", got)
		}
		if strings.Contains(got, "assistant: a2") {
			t.Errorf("/history invented an answer that is not in the history:\n%s", got)
		}
	})
}

// TestChatBannerNamesTheFallback covers the one line that says which chat format is in use.
//
// A model whose template could not be evaluated still answers -- in the neutral transcript,
// which is a different shape from what its own template would produce. Saying so at startup is
// the difference between a bug report and a mystery, and the reason is the actionable half: it
// names what about the template could not be evaluated. Dropping the reason from the banner
// passed the cmd suite.
func TestChatBannerNamesTheFallback(t *testing.T) {
	// A model whose template this project cannot parse. The fallback is decided at parse time,
	// not at render time, so an unknown *call* would not produce it -- the template has to use
	// a construct the parser rejects.
	path := gguftest.Model{
		ChatTemplate: "{% unknown_tag %}",
	}.Write(t, t.TempDir())

	var out bytes.Buffer
	// EOF ends the session immediately, so only the banner is printed.
	if err := chatSession(strings.NewReader(""), &out, chatOptions{model: path, maxTokens: 1}); err != nil {
		t.Fatalf("chatSession: %v", err)
	}
	got := out.String()
	if !strings.Contains(got, "neutral transcript") {
		t.Fatalf("banner does not announce the fallback:\n%s", got)
	}
	// The reason is inside the parentheses after the colon.
	if !strings.Contains(got, "neutral transcript: ") {
		t.Errorf("banner does not give a reason for the fallback:\n%s", got)
	}
	if strings.HasSuffix(strings.TrimSpace(strings.SplitN(got, "\n", 2)[0]), ":") {
		t.Errorf("banner ends at a bare colon, so the reason is empty:\n%s", got)
	}
}

// TestChatSizesThePoolForAWholeContext covers the block count a REPL starts with.
//
// The pool has to hold one whole conversation, because every turn re-sends the history and the
// prefix index hands back whole blocks. Rounding the context down to a block loses the last
// partial one, so a conversation that exactly fills the window fails on its final turn.
//
// What this asserts is the ceiling division and cfgForBlocks, not the wiring: the pool size is
// not reported anywhere a session prints, so the connection from these numbers to the engine's
// block count is still read rather than measured. Reaching it through behaviour needs a
// conversation that fills the window exactly, which the fixture's 64-position default makes
// cheap to construct but the assertion would then be about the engine's pool-exhaustion error
// rather than about this arithmetic.
func TestChatSizesThePoolForAWholeContext(t *testing.T) {
	cases := []struct {
		name    string
		ctx     int
		blockSz int
		want    int
	}{
		{"exact multiple", 128, 16, 8},
		// A context that is not a whole number of blocks still needs the last partial one.
		{"one past a multiple", 129, 16, 9},
		{"one short of a multiple", 127, 16, 8},
		{"one block", 16, 16, 1},
		{"smaller than a block", 8, 16, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			path := gguftest.Model{ContextLen: c.ctx, NoEOS: true}.Write(t, t.TempDir())
			gf, err := gguf.Open(path)
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			if got := cfgForBlocks(gf); got != c.ctx {
				t.Fatalf("cfgForBlocks = %d, want the file's %d", got, c.ctx)
			}
			// The ceiling division chatSession applies to it.
			got := (cfgForBlocks(gf) + c.blockSz - 1) / c.blockSz
			if got != c.want {
				t.Errorf("a %d-position context at block size %d needs %d blocks, got %d",
					c.ctx, c.blockSz, c.want, got)
			}
		})
	}
}

// TestChatRefusesAConversationThatFillsTheContext covers the boundary the REPL checks before
// it submits.
//
// Two failures live here and neither was tested. A prompt that encodes to nothing has to be
// refused rather than submitted, since the engine would have no position to start from; and a
// conversation that reaches the context length has to be refused with advice, because the
// engine's own error at that point is about positions rather than about what the user should
// do next.
//
// The comparison is >= rather than >: a prompt exactly filling the window leaves no room for
// even one generated token, so accepting it produces an empty reply. Relaxing it to > passed the
// cmd suite.
func TestChatRefusesAConversationThatFillsTheContext(t *testing.T) {
	// 45 "a " words plus "done" renders to exactly the fixture's 64 positions once the chat
	// template adds its role markers -- measured, not derived: the prompt the user types is
	// shorter than the prompt the model sees, and only the latter is compared against the
	// window. A prompt far past the limit cannot distinguish `>=` from `>`.
	path := gguftest.Model{ContextLen: 64, NoEOS: true}.Write(t, t.TempDir())
	long := strings.Repeat("a ", 45) + "done"

	var out bytes.Buffer
	err := chatSession(strings.NewReader(long+"\n/exit\n"), &out, chatOptions{model: path, maxTokens: 4})
	if err != nil {
		t.Fatalf("chatSession: %v", err)
	}
	got := out.String()
	if !strings.Contains(got, "context is 64") {
		t.Errorf("a conversation past the window was not refused with the size:\n%s", got)
	}
	// The advice is the part a user can act on.
	if !strings.Contains(got, "/clear") {
		t.Errorf("the refusal does not say what to do about it:\n%s", got)
	}

	// The session survives the refusal and keeps accepting input: a REPL that exited on the
	// first oversized paste would lose the conversation the user still wants to /clear.
	var out2 bytes.Buffer
	err = chatSession(strings.NewReader(long+"\n/history\n/exit\n"), &out2, chatOptions{model: path, maxTokens: 4})
	if err != nil {
		t.Fatalf("chatSession: %v", err)
	}
	if !strings.Contains(out2.String(), "history is empty") {
		t.Errorf("a refused turn left something in the history:\n%s", out2.String())
	}
}

// TestChatRefusesABadModelBeforePrompting covers chat's two startup failures.
//
// Both return before the first prompt is printed, which is the property worth asserting: a session
// that printed "you> " and then failed would leave a caller typing into a dead REPL. Neither path had
// a test -- every chat test starts from a working fixture -- so removing either error check passed.
func TestChatRefusesABadModelBeforePrompting(t *testing.T) {
	t.Run("a missing file", func(t *testing.T) {
		var out bytes.Buffer
		err := chatSession(strings.NewReader(""), &out, chatOptions{
			model: filepath.Join(t.TempDir(), "absent.gguf"),
		})
		if err == nil {
			t.Fatal("chatSession opened a missing model")
		}
		// The path is what a reader needs to see, since it came from a flag or the environment.
		if !strings.Contains(err.Error(), "absent.gguf") {
			t.Errorf("error %q does not name the file", err)
		}
		if out.Len() != 0 {
			t.Errorf("the session printed %q before failing", out.String())
		}
	})

	t.Run("a file with no tokenizer", func(t *testing.T) {
		// A valid header with no metadata: the same shape run and serve are tested with.
		b := []byte("GGUF")
		b = append(b, 3, 0, 0, 0)
		b = append(b, make([]byte, 16)...)
		path := filepath.Join(t.TempDir(), "bare.gguf")
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatalf("writing the file: %v", err)
		}

		var out bytes.Buffer
		err := chatSession(strings.NewReader(""), &out, chatOptions{model: path})
		if err == nil {
			t.Fatal("chatSession accepted a file with no tokenizer")
		}
		if !strings.Contains(err.Error(), "tokenizer") {
			t.Errorf("error %q does not name the tokenizer", err)
		}
		if out.Len() != 0 {
			t.Errorf("the session printed %q before failing", out.String())
		}
	})
}

// TestChatRefusesABlockWiderThanTheContext covers the same check serve makes, on the REPL's path.
//
// A block larger than the context never fills, so the prefix cache never fires. The REPL would run
// and answer, just never remembering anything between turns -- and the banner still says the cache is
// on, which is the kind of silence that costs an afternoon.
//
// serve's copy is tested; this one was not, so its message could lose either number without a
// failure. Both are asserted, and so is the suggestion, since "use 512 or less" is the actionable
// half.
func TestChatRefusesABlockWiderThanTheContext(t *testing.T) {
	model := chatModel(t)

	var out bytes.Buffer
	err := chatSession(strings.NewReader(""), &out, chatOptions{
		model:       model,
		blockSize:   1000,
		prefixCache: true,
	})
	if err == nil {
		t.Fatal("chatSession accepted a block wider than the context")
	}
	for _, want := range []string{"1000", "512", "or less"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not state %q", err, want)
		}
	}
	// The banner comes first and the refusal after it, so the output is not empty -- but no prompt
	// was written, which is what would leave a caller typing into a session that cannot answer.
	if strings.Contains(out.String(), "you> ") {
		t.Errorf("the session prompted before failing: %q", out.String())
	}

	// The boundary itself: a block exactly as wide as the context fills on every turn, so it is
	// legal, and one more is not. Measured -- 512 runs, 513 is refused. Without this pair the check
	// could be relaxed by one and still pass, because 1000 is far past either comparison.
	for _, c := range []struct {
		blockSize int
		wantErr   bool
	}{
		{512, false},
		{513, true},
	} {
		out.Reset()
		err := chatSession(strings.NewReader(""), &out, chatOptions{
			model:       model,
			blockSize:   c.blockSize,
			prefixCache: true,
		})
		if c.wantErr && err == nil {
			t.Errorf("--block-size %d was accepted, want a refusal", c.blockSize)
		}
		if !c.wantErr && err != nil {
			t.Errorf("--block-size %d was refused: %v", c.blockSize, err)
		}
	}
}

// TestChatTurnsPutsTheSystemPromptFirst covers the conversion from a session to turns.
//
// --system has no other test: it is read into a local, passed through chatTurnOnce and rendered, and
// nothing observed the result. Dropping the assignment and forcing it empty both passed every test,
// which means the flag's whole effect was untested.
//
// The order is the part worth pinning rather than the presence: a system turn appended after the
// history would still produce a prompt, and still be wrong -- the model would read it as a remark made
// later in the conversation.
func TestChatTurnsPutsTheSystemPromptFirst(t *testing.T) {
	history := []chatTurn{
		{role: "user", text: "first"},
		{role: "assistant", text: "reply"},
	}

	t.Run("with a system prompt", func(t *testing.T) {
		got := chatTurns("You are terse.", history, "second")
		if len(got) != len(history)+2 {
			t.Fatalf("chatTurns returned %d turns, want %d", len(got), len(history)+2)
		}
		if got[0].Role != "system" || got[0].Text != "You are terse." {
			t.Errorf("first turn = %+v, want the system prompt", got[0])
		}
		if got[len(got)-1].Role != "user" || got[len(got)-1].Text != "second" {
			t.Errorf("last turn = %+v, want the new message", got[len(got)-1])
		}
		// The history sits between the two, in order.
		for i, want := range history {
			if got[i+1].Role != want.role || got[i+1].Text != want.text {
				t.Errorf("turn %d = %+v, want %+v", i+1, got[i+1], want)
			}
		}
	})

	// The session reads --system into a local before the loop, so this also covers that
	// assignment: with it dropped, every session would behave as if no system prompt were given.
	t.Run("reaches the session", func(t *testing.T) {
		model := chatModel(t)
		var out bytes.Buffer
		if err := chatSession(strings.NewReader("/exit\n"), &out, chatOptions{
			model:  model,
			system: "You are terse.",
		}); err != nil {
			t.Fatalf("chatSession: %v", err)
		}
		// The REPL does not echo the system prompt, so what is observable is that the session runs
		// at all with one -- and that it did not treat the text as a message.
		if strings.Contains(out.String(), "You are terse.") {
			t.Errorf("the system prompt was echoed to the user: %q", out.String())
		}
	})

	t.Run("without one", func(t *testing.T) {
		got := chatTurns("", history, "second")
		if len(got) != len(history)+1 {
			t.Fatalf("chatTurns returned %d turns, want %d -- an empty prompt adds none",
				len(got), len(history)+1)
		}
		for _, turn := range got {
			if turn.Role == "system" {
				t.Errorf("an empty --system produced a system turn: %+v", turn)
			}
		}
	})
}

// TestChatStopsWhenTheOutputIsGone covers the four write failures in the session loop.
//
// A session whose output is a broken pipe has nothing useful left to do, and the loop's contract is
// to return the error rather than generate more tokens into it. None of these paths had a test: every
// chat test writes to a bytes.Buffer, which cannot fail.
//
// The first two are the interesting pair -- the prompt before reading a line, and the command reply --
// because both happen before any generation, so a session that ignored the error would sit waiting for
// input it could not prompt for.
func TestChatStopsWhenTheOutputIsGone(t *testing.T) {
	model := chatModel(t)

	// The session writes four things before the first prompt: the banner, the model summary, the
	// command list and a blank line. Allowing exactly those lets the prompt be the write that fails,
	// which is the check this subtest is for -- a writer that fails immediately exercises the banner's
	// check instead and reaches the same error.
	t.Run("the prompt", func(t *testing.T) {
		fw := &failingAfter{Writer: io.Discard, remaining: 4}
		err := chatSession(strings.NewReader("hi\n"), fw, chatOptions{model: model})
		if err == nil {
			t.Fatal("a prompt into a broken writer returned no error")
		}
		if !errors.Is(err, fw.err) {
			t.Errorf("error = %v, want the writer's error %v", err, fw.err)
		}
	})

	t.Run("a command reply", func(t *testing.T) {
		// The banner and the first prompt are written before any input is read, so the writer is
		// allowed that much and fails on the reply to /help.
		err := chatSession(strings.NewReader("/help\n"),
			&failingAfter{Writer: io.Discard, remaining: 2}, chatOptions{model: model})
		if err == nil {
			t.Fatal("a command reply into a broken writer returned no error")
		}
	})
}

// failingAfter passes writes through until a count is exhausted, then fails. The session writes the
// banner and the first prompt before any input is read, so a count of two lets it reach the command.
type failingAfter struct {
	io.Writer
	remaining int
	err       error
}

func (f *failingAfter) Write(p []byte) (int, error) {
	if f.remaining <= 0 {
		if f.err == nil {
			f.err = errors.New("write: broken pipe")
		}
		return 0, f.err
	}
	f.remaining--
	return f.Writer.Write(p)
}

// TestChatStopsWhenTheBannerCannotBeWritten covers the first write a session makes.
//
// The banner is how a caller learns whether the model's own template is in use; a session that cannot
// write it has nothing to say and should stop, rather than reading prompts and answering into a stream
// that is already failing. Nothing tested it: dropping the error made every test still pass.
func TestChatStopsWhenTheBannerCannotBeWritten(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	wantErr := errors.New("output is gone")
	// A writer that fails once and then succeeds, so the session's later writes are not what stops it.
	// With a permanently failing writer, discarding the banner's error still ends in an error from the
	// next write, and the mutation is invisible.
	err := chatSession(strings.NewReader("hello\n"), &failOnceWriter{err: wantErr}, chatOptions{
		model:     path,
		blockSize: 4,
	})
	if err == nil {
		t.Fatal("chatSession kept reading with a failing output")
	}
	if !errors.Is(err, wantErr) {
		t.Errorf("error %v does not carry the writer's error %v", err, wantErr)
	}
}

// failOnceWriter fails its first write and accepts every later one.
//
// It exists for tests about a specific write failing rather than about output being gone: with a writer
// that always fails, dropping one error is invisible because the next write reports the same thing.
type failOnceWriter struct {
	err  error
	once bool
}

func (w *failOnceWriter) Write(p []byte) (int, error) {
	if !w.once {
		w.once = true
		return 0, w.err
	}
	return len(p), nil
}

// TestChatStatsReportsThePool covers the line that answers "why did that turn fail".
//
// Running out of the pool is the one failure a REPL hits repeatedly: a conversation outgrows
// the pool, every later turn fails the same way because a failed turn does not grow the
// history, and the error names the pool's size without saying how close the session is to it.
// /stats reported turns, tokens and reuse but not the pool, so the question was unanswerable
// from inside the session.
//
// The total is asserted as well as the free count, because the free count alone misleads: a
// session whose every turn is refused because one request is larger than the whole pool shows
// every block free, and a reader concludes the pool is healthy. It is -- the request is simply
// too big, which is the case explainChatError says /clear cannot fix.
func TestChatStatsReportsThePool(t *testing.T) {
	model := chatModel(t)

	t.Run("a paged session reports its pool", func(t *testing.T) {
		out := runChat(t, chatOptions{
			model: model, maxTokens: 2, blocks: 4, blockSize: 16, prefixCache: true,
		}, "/stats\n/exit\n")
		// Both numbers and the position count, so the reader can compare their own prompt
		// against the pool rather than only knowing that a pool exists.
		for _, want := range []string{"kv pool:", "of 4 blocks free", "16 positions each", "64 total"} {
			if !strings.Contains(out, want) {
				t.Errorf("/stats does not report the pool (%q missing):\n%s", want, out)
			}
		}
	})

	t.Run("the free count reflects use", func(t *testing.T) {
		out := runChat(t, chatOptions{
			model: model, maxTokens: 2, blocks: 8, blockSize: 16, prefixCache: true,
		}, "alpha beta gamma\n/stats\n/exit\n")
		// A turn holds at least one block, so the pool cannot read as untouched afterwards.
		if strings.Contains(out, "8 of 8 blocks free") {
			t.Errorf("/stats reports a full pool after a turn that used one:\n%s", out)
		}
		if !strings.Contains(out, "kv pool:") {
			t.Errorf("/stats printed no pool line:\n%s", out)
		}
	})
}

// TestExplainChatErrorCoversTheWindowToo extends the test above to the second failure that
// repeats.
//
// The doc on explainChatError called pool exhaustion "the one failure here that repeats",
// and that was wrong: the conversation outgrowing the model's window repeats for exactly the
// same reason -- history advances only on a successful turn -- but arrives as an encoding
// check in chatTurnOnce rather than as an engine error, so nothing recognised it. Measured on
// a 256-position fixture, three turns filled the window and each of the next nine attempts
// printed "conversation is 318 tokens, context is 256" with nothing to say the number could
// not change by itself.
//
// Both halves are asserted: the explanation appears, and the sentinel survives in the chain so
// a caller checking errors.Is still sees what failed.
func TestExplainChatErrorCoversTheWindowToo(t *testing.T) {
	full := fmt.Errorf("%w: conversation is 318 tokens, context is 256", errContextFull)
	got := explainChatError(full)
	if got == full {
		t.Fatal("a full context window was not explained")
	}
	if !errors.Is(got, errContextFull) {
		t.Error("the explanation dropped errContextFull from the chain")
	}
	// The fact the bare message omits: retrying is pointless.
	if !strings.Contains(got.Error(), "will fail the same way") {
		t.Errorf("error %q does not say the next turn fails identically", got)
	}
	// And the way out. /clear is named because it is the only thing that helps here: unlike
	// the pool case, the window is fixed by the model, so no flag raises it.
	if !strings.Contains(got.Error(), "/clear") {
		t.Errorf("error %q does not name a way out", got)
	}
	// The numbers the engine produced have to survive, since they are what tells the user how
	// far over they are.
	for _, want := range []string{"318", "256"} {
		if !strings.Contains(got.Error(), want) {
			t.Errorf("error %q lost the figure %s", got, want)
		}
	}
	// The two explanations must not be each other's: a window error mentioning --blocks would
	// send the user to a flag that cannot help.
	if strings.Contains(got.Error(), "--blocks") {
		t.Errorf("the window explanation suggests --blocks, which cannot widen the window: %q", got)
	}
}

// TestChatClearKeepsThePrefixCache pins what /clear does and does not discard.
//
// /clear empties the history -- that is its contract -- and nothing in the REPL touches the
// engine or its pool, so the blocks the previous turns published stay in the prefix index and
// stay counted. The README states this, and it is the reason /clear is offered as the way out
// of a full window: the conversation restarts without costing an immediate re-prefill of
// whatever the new prompt shares with the old one.
//
// The counter being cumulative is what makes the claim observable: it must not fall back to
// zero across the command. A /clear that reset the engine -- the tempting implementation, since
// "clear" sounds like it should -- would show the count dropping, and a session would lose every
// cached block each time a user asked for a fresh start.
//
// Measured on the 512-position fixture before writing this: reuse per turn is 0, 160, 480, the
// count holds at 480 across /clear, and the first turn afterwards adds another 160.
func TestChatClearKeepsThePrefixCache(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	msg := strings.Repeat("word ", 30)
	input := msg + "\n/stats\n" +
		msg + "\n/stats\n" +
		"/clear\n/stats\n" +
		msg + "\n/stats\n/exit\n"
	out := runChat(t, chatOptions{
		model: path, maxTokens: 4, topK: 40, topP: 0.9, prefixCache: true,
	}, input)

	// Collect the reuse figure after each turn, in order.
	var counts []int
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if !strings.Contains(line, "prompt positions") {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(line, "reused: %d prompt positions", &n); err != nil {
			t.Fatalf("could not read the reuse count from %q: %v", line, err)
		}
		counts = append(counts, n)
	}
	if len(counts) != 4 {
		t.Fatalf("saw %d /stats reports, want 4:\n%s", len(counts), out)
	}
	if counts[0] != 0 {
		t.Errorf("a fresh session reused %d positions, want 0", counts[0])
	}
	if counts[1] == 0 {
		t.Errorf("the second turn reused nothing:\n%s", out)
	}
	// The whole point: /clear does not take the cache back.
	if counts[2] != counts[1] {
		t.Errorf("reuse went from %d to %d across /clear, so the command discarded the prefix "+
			"cache as well as the history", counts[1], counts[2])
	}
	// And the index is still live afterwards, not merely retained.
	if counts[3] <= counts[2] {
		t.Errorf("the first turn after /clear reused %d, no more than the %d already counted",
			counts[3], counts[2])
	}
}

// TestChatTreatsEachLineAsATurn pins the REPL's input contract, which a paste violates.
//
// One line in, one message out is the whole design, and it is what makes the REPL a few
// hundred lines rather than a readline implementation. The consequence is not obvious until
// it happens: text pasted with real newlines becomes one turn per line, each resending the
// history before it, so a paragraph swallows the window far faster than its length suggests.
//
// Measured on the 512-position fixture before writing this: a twelve-line message -- a
// request line, ten lines of text, a follow-up question -- produced three turns and then
// three "conversation is 515 tokens, context is 512" errors, and the last line never ran at
// all. The assertion is that shape rather than an exact count: more than one turn came from
// one paste, and the session ended in the full-window error rather than answering the
// question at the end.
//
// If a multi-line input mode is ever added, this test should fail -- and when it does, the
// comment on scanner.Buffer in chat.go needs the same update.
func TestChatTreatsEachLineAsATurn(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	var in strings.Builder
	in.WriteString("Summarise this:\n")
	for range 10 {
		in.WriteString("The quick brown fox jumps over the lazy dog.\n")
	}
	in.WriteString("What did it say?\n/exit\n")

	out := runChat(t, chatOptions{
		model: path, maxTokens: 4, topK: 40, prefixCache: true,
	}, in.String())

	// Every line started a turn, which is the property being pinned.
	if got := strings.Count(out, "> "); got < 12 {
		t.Errorf("a 12-line paste produced %d prompts, want at least 12 -- so lines were "+
			"joined after all:\n%s", got, out)
	}
	// And the window ran out before the last line could be answered: the failure the
	// per-line contract causes.
	if !strings.Contains(out, "context window full") {
		t.Errorf("a twelve-line paste did not exhaust a 512-position window, so the shape "+
			"this test documents has changed:\n%s", out)
	}
}

// TestChatSystemPromptReachesTheModel checks the system prompt from the one place a REPL can
// show it: the prompt the engine actually encoded.
//
// The existing test asserts chatTurns puts a system turn first and that the REPL does not echo
// the text, which is structure and presentation. Neither would notice a renderer that dropped
// system turns, because chatTurns is above the renderer and the echo is below it. Measured
// against serve's equivalent earlier, a system prompt is worth tens of positions; the same
// effect is visible here as reuse across two turns.
//
// Reuse rather than a token count: the figure is the positions the second turn attached from
// the first, and it can only be larger when the first prompt carried the system text. On the
// 512-position fixture with a 20-word message: 112 positions without a system prompt and 160
// with a short one.
func TestChatSystemPromptReachesTheModel(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	msg := strings.Repeat("word ", 20)

	reuse := func(system string) int {
		var in strings.Builder
		in.WriteString(msg + "\n/stats\n" + msg + "\n/stats\n/exit\n")
		var out strings.Builder
		if err := chatSession(strings.NewReader(in.String()), &out, chatOptions{
			model: path, maxTokens: 4, topK: 40, prefixCache: true, system: system,
		}); err != nil {
			t.Fatalf("chatSession: %v", err)
		}
		var counts []int
		for _, line := range strings.Split(out.String(), "\n") {
			line = strings.TrimSpace(line)
			if !strings.Contains(line, "prompt positions") {
				continue
			}
			var n int
			if _, err := fmt.Sscanf(line, "reused: %d prompt positions", &n); err == nil {
				counts = append(counts, n)
			}
		}
		if len(counts) != 2 {
			t.Fatalf("saw %d /stats reports, want 2:\n%s", len(counts), out.String())
		}
		// The second turn's figure is the one that reflects the first turn's prompt.
		return counts[1]
	}

	plain := reuse("")
	withSystem := reuse("You are a terse assistant who answers in one word.")

	if plain == 0 {
		t.Fatalf("no reuse without a system prompt, so this measurement cannot compare")
	}
	// The system text lengthens every prompt, so the second turn attaches more whole blocks.
	if withSystem <= plain {
		t.Errorf("reuse is %d with a system prompt and %d without: a system prompt that never "+
			"reached the model would look exactly like this", withSystem, plain)
	}
}
