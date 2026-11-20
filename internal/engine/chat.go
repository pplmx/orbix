package engine

import (
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/pplmx/orbix/internal/chat"
	"github.com/pplmx/orbix/internal/gguf"
)

// Turn is one message in a chat conversation.
//
// It lives in engine because the prompt a conversation turns into is a property of
// serving the model, not of any one front end: the REPL and the HTTP server must
// render the same transcript for the same conversation, or the same exchange produces
// different tokens depending on how it was asked.
type Turn struct {
	// Role is who spoke: "system", "user", or "assistant". An empty role is treated
	// as "user", which is what every chat API does with a missing one.
	Role string
	// Text is the message body.
	Text string
}

// RenderChatPrompt renders a conversation as a neutral "Role: content" transcript.
//
// It is the fallback, not the default: ChatRenderer renders a checkpoint's own template
// from tokenizer.chat_template whenever that template falls inside internal/chat's
// supported subset, and calls this only when it does not. The neutral form is what a
// model without a usable template gets -- a base model, or one whose template uses a
// construct this project has not implemented.
//
// Producing a model's markers for a model that was not trained on them would be worse
// than a neutral transcript, because the markers move it off the distribution it expects.
// That is why the two paths exist separately rather than one template approximating the
// other.
func RenderChatPrompt(turns []Turn) string {
	var b strings.Builder
	for _, t := range turns {
		role := t.Role
		if role == "" {
			role = "user"
		}
		// Capitalize the first character, not the first byte. role comes from the client's
		// JSON, so it can be any string: slicing [:1] cuts a multi-byte rune in half, and the
		// transcript then shows U+FFFD followed by the stray continuation byte. Measured with
		// role "é": "\ufffd\xa9: x". Decoding the first rune and keeping the rest as text is
		// what makes the capitalizing independent of the encoding.
		if first, size := utf8.DecodeRuneInString(role); first != utf8.RuneError || size > 1 {
			b.WriteString(strings.ToUpper(string(first)))
			b.WriteString(role[size:])
		} else {
			b.WriteString(role)
		}
		b.WriteString(": ")
		b.WriteString(strings.TrimSpace(t.Text))
		b.WriteString("\n")
	}
	b.WriteString("Assistant:")
	return b.String()
}

// ChatRenderer renders conversations with a checkpoint's own chat template when that
// template is one this project can evaluate, and with a neutral transcript otherwise.
//
// The template is a property of the model, so the renderer is built once per model
// rather than per request. It records why it fell back, because "the model's format was
// not used" is something an operator needs to see: the server prints it at startup and
// the REPL shows it on the first turn.
//
// It is safe for concurrent use, which the server relies on: one renderer serves every
// request goroutine. Render reads the configuration fields that are set once at
// construction and takes no lock for them; the mutex below covers renderReason alone,
// which is the one field a request writes. Anything added to this type has to join one
// group or the other, and the accessors that read renderReason already lock.
type ChatRenderer struct {
	tmpl *chat.Template
	// fallbackReason is empty when the model's template is in use.
	fallbackReason string
	// mu guards renderReason, which Render sets from whichever goroutine is serving a
	// request. It holds the first render failure, or "" while rendering succeeds.
	mu           sync.Mutex
	renderReason string
	// bos and eos are the vocabulary's special pieces as text, which a template
	// interpolates.
	bos, eos string
}

// NewChatRenderer reads tokenizer.chat_template from a GGUF file and compiles it.
//
// A model without a template, or with one using a construct outside the supported
// subset, gets the neutral transcript and a reason. That is deliberate: a partially
// rendered template would produce a prompt that looks plausible and is wrong, while a
// neutral transcript is merely less good than the model's own format.
func NewChatRenderer(f *gguf.File) (*ChatRenderer, error) {
	r := &ChatRenderer{}
	r.bos, r.eos = specialPieceText(f, "tokenizer.ggml.bos_token_id"), specialPieceText(f, "tokenizer.ggml.eos_token_id")

	src, ok := f.Get[string]("tokenizer.chat_template")
	if !ok || strings.TrimSpace(src) == "" {
		r.fallbackReason = "the model declares no tokenizer.chat_template"
		return r, nil
	}
	tmpl, err := chat.Parse(src)
	if err != nil {
		r.fallbackReason = err.Error()
		return r, nil
	}
	r.tmpl = tmpl
	return r, nil
}

// UsesModelTemplate reports whether the model's own format is in use.
//
// A template that failed to render is not in use: the prompts it produces are the neutral
// transcript. Reporting true there would tell a caller its sessions use the model's format
// when every one of them has fallen back.
func (r *ChatRenderer) UsesModelTemplate() bool {
	if r == nil || r.tmpl == nil {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.renderReason == ""
}

// HasTemplate reports whether the model declared a template this project could parse.
//
// It is not the same question as UsesModelTemplate, and the REPL needs both. A model with no
// template at all is in the fallback from the first turn and said so in the banner; a model
// whose template parses but fails to render *this conversation* starts in the model's format
// and changes to the fallback partway through. Reporting the second case as if it were the
// first is what the REPL did: it printed "the model's template could not render this
// conversation" after the very first successful turn on a model that declares no template,
// contradicting the banner above it.
func (r *ChatRenderer) HasTemplate() bool {
	return r != nil && r.tmpl != nil
}

// FallbackReason explains why the model's format is not in use, or is empty when it is.
func (r *ChatRenderer) FallbackReason() string {
	if r == nil {
		return ""
	}
	if r.fallbackReason != "" {
		return r.fallbackReason
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.renderReason
}

// Render produces the prompt for a conversation.
//
// A template that compiles can still fail at render time -- an undefined variable in a
// branch this conversation takes, for instance -- so a render error also falls back
// rather than failing the request. FallbackReason records why.
//
// There is no error return because there is no failing path: every branch here produces a
// prompt. An error that is always nil reads as a case the caller has to handle, and both
// callers either checked it or discarded it with _, neither of which could ever matter.
func (r *ChatRenderer) Render(turns []Turn) string {
	if r == nil || r.tmpl == nil {
		return RenderChatPrompt(turns)
	}
	msgs := make([]chat.Message, len(turns))
	for i, t := range turns {
		role := t.Role
		if role == "" {
			role = "user"
		}
		msgs[i] = chat.Message{Role: role, Content: strings.TrimSpace(t.Text)}
	}
	out, err := r.tmpl.Render(msgs, chat.Inputs{
		BOSToken:            r.bos,
		EOSToken:            r.eos,
		AddGenerationPrompt: true,
	})
	if err != nil {
		// The template parsed but cannot render this conversation. Falling back keeps the
		// session usable, but the reason has to be recorded: the alternative is a prompt in
		// the neutral format while the caller reports the model's own template, which is
		// how a multi-turn Qwen2 conversation ended up with "Human:" in it and no
		// indication why.
		r.mu.Lock()
		r.renderReason = err.Error()
		r.mu.Unlock()
		return RenderChatPrompt(turns)
	}
	return out
}

// specialPieceText returns the surface text of a token id, or the empty string.
//
// A template interpolates bos_token as text rather than as an id, so the piece has to be
// looked up in the vocabulary; getting it wrong prepends something the model reads as
// content.
func specialPieceText(f *gguf.File, key string) string {
	id, ok := f.Get[int](key)
	if !ok || id < 0 {
		return ""
	}
	pieces, ok := f.Strings("tokenizer.ggml.tokens")
	if !ok || id >= len(pieces) {
		return ""
	}
	return pieces[id]
}
