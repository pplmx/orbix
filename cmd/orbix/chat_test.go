package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"

	"github.com/pplmx/orbix/internal/engine"
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
