package engine

import (
	"fmt"
	"maps"
	"slices"
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
	// ToolCalls are the calls an assistant turn made, in the order the model made them.
	//
	// A template renders them into the assistant turn so the model reads back the call it
	// made before it reads the result; the neutral transcript has no place for them and
	// prints Text alone, which is why a conversation carrying calls on a model with no usable
	// template loses them rather than inventing a format.
	ToolCalls []ToolCall
}

// ToolCall is one function call in a conversation, as a template sees it.
//
// It mirrors chat.ToolCall rather than re-exporting it, for the same reason Turn mirrors
// chat.Message: the front ends below this layer should not have to import the template
// evaluator to describe a conversation.
type ToolCall struct {
	// ID is the identifier the response assigned. It is what a tool result names when it
	// answers this call, and a template that numbers calls itself ignores it.
	ID string
	// Name is the function that was called.
	Name string
	// Arguments is the call's arguments as the JSON object text the model produced, kept
	// verbatim so the prompt carries the model's own spelling rather than a re-encoding.
	Arguments string
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
		// The parser's message is about its own grammar ("expected \",\" or \")\" in call to
		// \"namespace\", got \"=\""), which names the construct at fault but not what it means
		// for the caller. Appending the consequence is what turns it into something an operator
		// can act on: the server still answers, in a format the model was not trained on.
		//
		// Measured on Qwen3-0.6B's own template, which uses namespace(multi_step_tool=...): the
		// neutral transcript answers a question correctly and then continues inventing turns,
		// because that is not the shape the model saw in training.
		r.fallbackReason = err.Error() +
			" (this template uses a construct outside the supported Jinja subset, so prompts " +
			"use a neutral transcript instead of the model's own format)"
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
	out, _ := r.RenderWith(turns, nil)
	return out
}

// RenderWith renders a conversation with extra template variables, returning an error for a
// variable this model cannot honour.
//
// Unlike a render failure, which falls back inside this call, an unusable variable is reported:
// falling back would answer with a prompt that silently ignored what the caller asked for, which
// is the one outcome a caller cannot detect from the reply. The three ways a variable is
// unusable are all decidable before rendering:
//
//   - the model declares no template this project can evaluate, so no variable reaches anything;
//   - the name is one the renderer binds itself (messages, bos_token, ...) -- chat refuses it;
//   - the template never names it, so binding it changes nothing.
//
// The third is a judgement rather than a fact, and it is the reason this lives here instead of
// in internal/chat: a caller rendering for its own purposes may well pass a variable on the
// chance that a template reads it, while a network endpoint has to answer for every field it
// accepts. internal/server refuses; the REPL passes none.
func (r *ChatRenderer) RenderWith(turns []Turn, vars map[string]any) (string, error) {
	return r.RenderConversation(turns, nil, vars)
}

// RenderConversation renders a conversation that may declare tools.
//
// Tools are separate from vars because they are not a template variable a caller gets to set:
// they are the list the request declared, and internal/chat binds them as `tools` so a template's
// tool-calling branch can print each signature. A model without a usable template has nowhere to
// put them, and that is refused rather than dropped -- see the caller.
func (r *ChatRenderer) RenderConversation(turns []Turn, tools []ToolSpec, vars map[string]any) (string, error) {
	if len(tools) > 0 && (r == nil || r.tmpl == nil) {
		return "", fmt.Errorf("engine: cannot declare tools, because this model declares no chat " +
			"template this project can evaluate; its calls would reach no prompt and the model " +
			"would answer as though it had never been told the tools exist")
	}
	if len(vars) > 0 {
		if r == nil || r.tmpl == nil {
			// No "engine:" prefix: the server turns this straight into the message of a 400,
			// so the prefix would be the only place in that response naming an internal
			// package. Measured against a live server, every other refusal in the same
			// envelope reads as a sentence about the request.
			return "", fmt.Errorf("cannot set template variables (%s), because this model "+
				"declares no chat template this project can evaluate", strings.Join(slices.Sorted(maps.Keys(vars)), ", "))
		}
		for _, name := range slices.Sorted(maps.Keys(vars)) {
			if !r.tmpl.Reads(name) {
				return "", fmt.Errorf("this model's chat template never reads %q, so "+
					"setting it would change nothing about the prompt", name)
			}
		}
	}
	if r == nil || r.tmpl == nil {
		return RenderChatPrompt(turns), nil
	}
	msgs := make([]chat.Message, len(turns))
	for i, t := range turns {
		role := t.Role
		if role == "" {
			role = "user"
		}
		calls := make([]chat.ToolCall, len(t.ToolCalls))
		for j, c := range t.ToolCalls {
			calls[j] = chat.ToolCall{ID: c.ID, Name: c.Name, Arguments: c.Arguments}
		}
		// Content is trimmed; the arguments are not. A tool call's JSON is machine text the
		// template re-emits, and trimming it would change the prompt the model reads back.
		msgs[i] = chat.Message{Role: role, Content: strings.TrimSpace(t.Text), ToolCalls: calls}
	}
	out, err := r.tmpl.Render(msgs, chat.Inputs{
		BOSToken:            r.bos,
		EOSToken:            r.eos,
		AddGenerationPrompt: true,
		Tools:               chatTools(tools),
		Variables:           vars,
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
		return RenderChatPrompt(turns), nil
	}
	return out, nil
}

// ToolSpec is one function a request declares, in the shape a chat template prints.
//
// It is the model-facing description rather than the request's own JSON: Name, Description and
// Parameters are the three fields every published template reads, and Parameters is the JSON
// schema as text because a template only ever re-emits it.
type ToolSpec struct {
	Name        string
	Description string
	// Parameters is the JSON schema object, as text. Empty is allowed -- a tool with no
	// arguments is a real shape -- and renders as an empty value rather than as an error.
	Parameters string
}

// chatTools converts the tool list for the template evaluator.
func chatTools(specs []ToolSpec) []chat.Tool {
	if len(specs) == 0 {
		return nil
	}
	out := make([]chat.Tool, len(specs))
	for i, t := range specs {
		out[i] = chat.Tool{Name: t.Name, Description: t.Description, Parameters: t.Parameters}
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
