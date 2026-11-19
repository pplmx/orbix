package chat

import "testing"

// FuzzParse checks that no template source makes the parser panic.
//
// The source comes from a checkpoint's tokenizer.chat_template, which is a string in a file the
// user downloaded. A panic here takes the process down while it is loading a model, and the
// failure would look like a broken build rather than a bad file. The parser is written to
// report what it does not understand, so the property to hold is that it always reports.
//
// Rendering is fuzzed too, on a template that parsed: a syntactically valid template can still
// drive the evaluator into an empty sequence, a missing field or a loop over the wrong shape,
// and those paths are reachable from a file as well.
func FuzzParse(f *testing.F) {
	seeds := []string{
		"",
		"plain text",
		"{{ 'a' }}",
		"{% if x %}a{% endif %}",
		"{% for m in messages %}{{ m.role }}{% endfor %}",
		"{% set x = 'a' + 'b' %}{{ x }}",
		"{{ messages[0]['content'] | trim }}",
		"{%- if messages[0]['role'] == 'system' -%}{{ messages[0]['content'] }}{%- endif -%}",
		"{{",
		"{%",
		"{{ x | }}",
		"{% if %}",
		"{% for x in %}{% endfor %}",
		"{{ x[ }}",
		"{{ a.b.c }}",
		"{% filter upper %}x{% endfilter %}",
		"{% macro m() %}x{% endmacro %}",
		"{{ 'a' if b else 'c' }}",
		"{% set ns = namespace(x=1) %}",
		"{{ tools | tojson(indent=4) }}",
		"{% if x is defined and x %}y{% endif %}",
		"{{ 'a' in 'abc' }}",
		"{% for k, v in tools %}{{ k }}{% endfor %}",
		"{{ a = b }}",
		"\x00\xff",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, src string) {
		// The input is wider than production ever supplies. A template reaches this package
		// through gguf's string reader, which refuses a value that is not valid UTF-8, so the
		// source here is always well-formed in reality. Fuzzing with arbitrary bytes is still
		// worth it -- the parser should not panic on anything -- but it means the rendered
		// output can be invalid UTF-8, and that is not a defect this target may report:
		// measured, "{{ '\xff\xfe' }}" parses and renders the two bytes back unchanged, and
		// the same source cannot arrive from a file.
		tmpl, err := Parse(src)
		if err != nil {
			// A reported error is the designed answer for anything outside the subset.
			return
		}
		// A template that parsed has to render without panicking, whatever it is asked to
		// render. The inputs exercise the branches a real conversation would, including the
		// empty one.
		_, _ = tmpl.Render(nil, Inputs{})
		_, _ = tmpl.Render([]Message{{Role: "user", Content: "hi"}}, Inputs{AddGenerationPrompt: true})
		_, _ = tmpl.Render(
			[]Message{{Role: "system", Content: "s"}, {Role: "user", Content: "u"}},
			Inputs{Tools: []Tool{{Name: "t", Description: "d", Parameters: `{}`}},
				BOSToken: "<s>", EOSToken: "</s>"},
		)
	})
}
