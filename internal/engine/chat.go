package engine

import "strings"

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

// RenderChatPrompt renders a conversation as the plain-text transcript this project
// uses as its chat template.
//
// The template is deliberately neutral -- "Role: content" lines with the assistant's
// turn left open -- rather than any model's real one. GGUF stores a model's template
// as a Jinja2 string, and evaluating Jinja is a dependency this project will not take;
// producing a Llama checkpoint's "<|start_header_id|>" markers for a model that was
// not trained on them would be worse than producing a neutral transcript, because the
// markers move the model off the distribution it expects.
//
// A caller that needs a model's exact template renders it itself and sends the result
// to the completions endpoint, which takes text verbatim.
func RenderChatPrompt(turns []Turn) string {
	var b strings.Builder
	for _, t := range turns {
		role := t.Role
		if role == "" {
			role = "user"
		}
		b.WriteString(strings.ToUpper(role[:1]))
		b.WriteString(role[1:])
		b.WriteString(": ")
		b.WriteString(strings.TrimSpace(t.Text))
		b.WriteString("\n")
	}
	b.WriteString("Assistant:")
	return b.String()
}
