package main

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/llama"
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
	for _, want := range []string{"Commands:", "/clear", `unknown command "/nope"`} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
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
