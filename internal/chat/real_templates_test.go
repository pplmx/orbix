package chat

import (
	"strings"
	"testing"
)

// These are the templates of two widely used checkpoints, copied from their
// tokenizer_config.json. They are the reason the supported subset is what it is: each
// construct here was added because one of these needed it, and a change that breaks
// either is caught without downloading a model.

// qwen2Template is Qwen2.5-Instruct's template, minus the tool-calling branch's
// selectattr/map usage, which this package does not implement. The tools branch itself
// is kept because it exercises tojson and a nested for.
const qwen2Template = `{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\n" }}
    {%- for tool in tools %}
        {{- tool | tojson }}
    {%- endfor %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}`

// llama3Template is Llama 3.1-Instruct's template in full.
const llama3Template = `{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}`

// TestQwen2TemplateRenders checks the exact prompt Qwen2 expects, including the default
// system turn it inserts when the conversation has none.
func TestQwen2TemplateRenders(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "What is 2+2?"}}
	got := render(t, qwen2Template, msgs, Inputs{AddGenerationPrompt: true})
	want := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
		"<|im_start|>user\nWhat is 2+2?<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Errorf("got  %q\nwant %q", got, want)
	}
}

// TestQwen2TemplateUsesSystemMessage checks the branch that matters when a caller does
// supply a system prompt: the model's own default must not appear as well.
func TestQwen2TemplateUsesSystemMessage(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "You are terse."},
		{Role: "user", Content: "hi"},
	}
	got := render(t, qwen2Template, msgs, Inputs{AddGenerationPrompt: false})
	want := "<|im_start|>system\nYou are terse.<|im_end|>\n" +
		"<|im_start|>user\nhi<|im_end|>\n"
	if got != want {
		t.Errorf("got  %q\nwant %q", got, want)
	}
}

// TestLlama3TemplateRenders checks Llama 3's format, which uses a different marker set
// and a trim filter on the content.
func TestLlama3TemplateRenders(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "  hi  "}}
	got := render(t, llama3Template, msgs, Inputs{BOSToken: "<|begin_of_text|>"})
	want := "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>" +
		"<|start_header_id|>assistant<|end_header_id|>\n\n"
	if got != want {
		t.Errorf("got  %q\nwant %q", got, want)
	}
}

// TestRealTemplatesRejectNothing pins the parse step separately: a template that
// compiles cannot later surprise the renderer with an unsupported construct.
func TestRealTemplatesRejectNothing(t *testing.T) {
	for name, src := range map[string]string{"qwen2": qwen2Template, "llama3": llama3Template} {
		if _, err := Parse(src); err != nil {
			t.Errorf("%s template failed to parse: %v", name, err)
		}
	}
}

// TestQwen2TemplateWithTools exercises the tools branch, which renders a JSON array and
// uses a nested loop.
func TestQwen2TemplateWithTools(t *testing.T) {
	msgs := []Message{{Role: "system", Content: "you can call tools"}, {Role: "user", Content: "go"}}
	got := render(t, qwen2Template, msgs, Inputs{AddGenerationPrompt: false, Tools: []Tool{
		{Name: "get", Description: "fetch", Parameters: `{}`},
	}})
	if !strings.Contains(got, "# Tools") {
		t.Errorf("prompt does not contain the tools heading: %q", got)
	}
	if !strings.Contains(got, `{"function"`) {
		t.Errorf("prompt does not contain the serialized tool: %q", got)
	}
}
