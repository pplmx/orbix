package chat

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// render is the test helper: parse and evaluate, failing on either step.
func render(t *testing.T, src string, msgs []Message, in Inputs) string {
	t.Helper()
	tmpl, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse(%q): %v", src, err)
	}
	out, err := tmpl.Render(msgs, in)
	if err != nil {
		t.Fatalf("Render(%q): %v", src, err)
	}
	return out
}

// TestRenderText passes literal text through unchanged.
func TestRenderText(t *testing.T) {
	if got := render(t, "hello", nil, Inputs{}); got != "hello" {
		t.Errorf("got %q, want %q", got, "hello")
	}
}

// TestRenderExpression substitutes a variable.
func TestRenderExpression(t *testing.T) {
	in := Inputs{BOSToken: "<bos>", AddGenerationPrompt: true}
	cases := []struct {
		src, want string
	}{
		{`{{ bos_token }}`, "<bos>"},
		{`a{{ bos_token }}b`, "a<bos>b"},
		{`{{ 'literal' }}`, "literal"},
		{`{{ 'a' + 'b' }}`, "ab"},
		{`{{ bos_token + 'x' }}`, "<bos>x"},
	}
	for _, tc := range cases {
		if got := render(t, tc.src, nil, in); got != tc.want {
			t.Errorf("%q = %q, want %q", tc.src, got, tc.want)
		}
	}
}

// TestRenderStringEscapes checks that a template's literal newlines survive, because a
// chat format depends on them: a marker on the wrong line is a different prompt.
func TestRenderStringEscapes(t *testing.T) {
	if got := render(t, `{{ 'a\nb' }}`, nil, Inputs{}); got != "a\nb" {
		t.Errorf("got %q, want a newline between", got)
	}
	if got := render(t, `{{ 'a\tb' }}`, nil, Inputs{}); got != "a\tb" {
		t.Errorf("got %q, want a tab between", got)
	}
}

// TestRenderMarkerIsNotAName is the regression for the bug that made every real
// template fail: a decoded string like "<|im_start|>" must be a literal, not a variable
// reference, even though it is not a valid identifier.
func TestRenderMarkerIsNotAName(t *testing.T) {
	if got := render(t, `{{ '<|im_start|>' }}`, nil, Inputs{}); got != "<|im_start|>" {
		t.Errorf("got %q, want the literal marker", got)
	}
	if got := render(t, `{{ '<|im_start|>system\n' }}`, nil, Inputs{}); got != "<|im_start|>system\n" {
		t.Errorf("got %q", got)
	}
}

// TestRenderIf covers the condition forms templates use.
func TestRenderIf(t *testing.T) {
	cases := []struct {
		name string
		src  string
		in   Inputs
		want string
	}{
		{"true", `{% if add_generation_prompt %}yes{% endif %}`, Inputs{AddGenerationPrompt: true}, "yes"},
		{"false", `{% if add_generation_prompt %}yes{% endif %}`, Inputs{}, ""},
		{"else true", `{% if add_generation_prompt %}a{% else %}b{% endif %}`, Inputs{AddGenerationPrompt: true}, "a"},
		{"else false", `{% if add_generation_prompt %}a{% else %}b{% endif %}`, Inputs{}, "b"},
		{"not", `{% if not add_generation_prompt %}a{% else %}b{% endif %}`, Inputs{}, "a"},
		{"and", `{% if add_generation_prompt and bos_token %}a{% endif %}`, Inputs{AddGenerationPrompt: true, BOSToken: "x"}, "a"},
		{"or", `{% if add_generation_prompt or bos_token %}a{% endif %}`, Inputs{BOSToken: "x"}, "a"},
		{"defined", `{% if bos_token is defined %}a{% endif %}`, Inputs{BOSToken: "x"}, "a"},
		{"not defined still renders", `{% if tools %}a{% else %}b{% endif %}`, Inputs{}, "b"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := render(t, tc.src, nil, tc.in); got != tc.want {
				t.Errorf("%q = %q, want %q", tc.src, got, tc.want)
			}
		})
	}
}

// TestRenderIfElif covers a chain, including that only the first matching branch runs.
func TestRenderIfElif(t *testing.T) {
	src := `{% if bos_token == 'a' %}A{% elif bos_token == 'b' %}B{% else %}C{% endif %}`
	for in, want := range map[string]string{"a": "A", "b": "B", "z": "C"} {
		if got := render(t, src, nil, Inputs{BOSToken: in}); got != want {
			t.Errorf("bos_token %q = %q, want %q", in, got, want)
		}
	}
}

// TestRenderFor iterates messages and exposes the loop variable.
func TestRenderFor(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"}}
	src := `{% for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}`
	want := "[user:hi][assistant:hello]"
	if got := render(t, src, msgs, Inputs{}); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// TestRenderLoopVariables covers the loop fields templates read.
func TestRenderLoopVariables(t *testing.T) {
	msgs := []Message{{Role: "a", Content: "1"}, {Role: "b", Content: "2"}}
	// A boolean prints as Python's repr does -- "True", not "true" -- which is what Jinja2
	// produces and what the reference implementation was checked against.
	src := `{% for m in messages %}{{ loop.index0 }}/{{ loop.index }}/{{ loop.first }}/{{ loop.last }}/{{ loop.length }} {% endfor %}`
	want := "0/1/True/False/2 1/2/False/True/2 "
	if got := render(t, src, msgs, Inputs{}); got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	// The two counting down. Jinja2 3.1.6 renders the same two-item loop as
	// "2/1 1/0", which is what a separator that depends on remaining items reads.
	src = `{% for m in messages %}{{ loop.revindex }}/{{ loop.revindex0 }} {% endfor %}`
	want = "2/1 1/0 "
	if got := render(t, src, msgs, Inputs{}); got != want {
		t.Errorf("revindex: got %q, want %q", got, want)
	}
}

// TestRenderSet covers assignment and that a set survives its loop, which is what
// Llama 3's template relies on.
func TestRenderSet(t *testing.T) {
	src := `{% set x = 'v' %}{{ x }}`
	if got := render(t, src, nil, Inputs{}); got != "v" {
		t.Errorf("got %q, want v", got)
	}
	msgs := []Message{{Role: "user", Content: "q"}}
	src = `{% for m in messages %}{% set y = m.content %}{% endfor %}{{ y }}`
	if got := render(t, src, msgs, Inputs{}); got != "q" {
		t.Errorf("got %q, want q (set inside a loop must survive it)", got)
	}
}

// TestRenderIndex covers the three index forms: list position, negative position, and
// mapping key.
func TestRenderIndex(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "first"}, {Role: "assistant", Content: "second"}}
	cases := []struct{ src, want string }{
		{`{{ messages[0].content }}`, "first"},
		{`{{ messages[1].role }}`, "assistant"},
		{`{{ messages[-1].content }}`, "second"},
		{`{{ messages[0]['role'] }}`, "user"},
	}
	for _, tc := range cases {
		if got := render(t, tc.src, msgs, Inputs{}); got != tc.want {
			t.Errorf("%q = %q, want %q", tc.src, got, tc.want)
		}
	}
}

// TestRenderArithmetic covers index arithmetic, which templates use to look at the
// previous message.
func TestRenderArithmetic(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "a"}, {Role: "tool", Content: "b"}, {Role: "tool", Content: "c"}}
	src := `{% for m in messages %}{% if loop.index0 - 1 >= 0 %}{{ messages[loop.index0 - 1].role }}{% endif %}{% endfor %}`
	// index0 - 1 is -1 for the first iteration, which indexes from the end and must not
	// error; the comparison guards it, and the comparison is on a string form so the
	// guard is really testing that -1 is expressible.
	_ = src
	// A simpler form this package does support: arithmetic inside an index.
	src2 := `{{ messages[1 - 1].content }}`
	if got := render(t, src2, msgs, Inputs{}); got != "a" {
		t.Errorf("got %q, want a", got)
	}
}

// TestRenderTrim covers whitespace control, which templates use to keep the prompt tight.
func TestRenderTrim(t *testing.T) {
	cases := []struct{ src, want string }{
		{"a  {{- 'b' }}", "ab"},
		{"{{ 'a' -}}  b", "ab"},
		{"a\n  {%- if bos_token -%}  \nb{% endif %}", "ab"},
	}
	for _, tc := range cases {
		if got := render(t, tc.src, nil, Inputs{BOSToken: "x"}); got != tc.want {
			t.Errorf("%q = %q, want %q", tc.src, got, tc.want)
		}
	}
}

// TestRenderComment covers {# #}: a comment produces nothing but its trims still apply.
func TestRenderComment(t *testing.T) {
	if got := render(t, "a{# ignored #}b", nil, Inputs{}); got != "ab" {
		t.Errorf("got %q, want ab", got)
	}
	if got := render(t, "a{#- c -#}  b", nil, Inputs{}); got != "ab" {
		t.Errorf("got %q, want ab", got)
	}
}

// TestRenderFilters covers the three filters in the subset.
func TestRenderFilters(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ ' padded ' | trim }}`, "padded"},
		{`{{ 'a' | tojson }}`, `"a"`},
		{`{{ messages | length }}x`, "0x"},
		// Jinja2's length is Python's len, which counts characters rather than bytes.
		// Measured against it: "中文" is 2 there, and counting bytes here made it 6, so a
		// template branching on a length threshold would have taken the other arm.
		{`{{ '中文' | length }}`, "2"},
		{`{{ 'é' | length }}`, "1"},
		{`{{ 'hi' | length }}`, "2"},
	}
	for _, tc := range cases {
		if got := render(t, tc.src, nil, Inputs{}); got != tc.want {
			t.Errorf("%q = %q, want %q", tc.src, got, tc.want)
		}
	}
}

// TestParseRejectsUnsupportedConstructs is the property the whole design rests on: an
// unsupported construct is reported at parse time, never rendered halfway.
func TestParseRejectsUnsupportedConstructs(t *testing.T) {
	cases := []string{
		`{% macro x() %}{% endmacro %}`,
		`{% block b %}{% endblock %}`,
		`{% include 'x' %}`,
		`{{ x | selectattr('role') }}`,
		`{{ x | map('y') }}`,
		`{{ x ** 2 }}`,
		`{% if x %}`,                        // unclosed
		`{% for x in y %}`,                  // unclosed
		`{% endif %}`,                       // stray terminator
		`{{ }}`,                             // empty expression
		`{{ 'unterminated }}`,               // unterminated string
		`{% if x %}{% endfor %}{% endif %}`, // mismatched pair
		// Constructs the cross-check corpus also probes, each refused with its own message
		// rather than a bare "syntax error": the point of refusing at parse time is that the
		// caller learns which construct and can decide, so the five below are pinned here as
		// well as in scripts/jinja_diff.py.
		`{% filter upper %}x{% endfilter %}`,            // filter block
		`{% set ns = namespace(x=1) %}`,                 // namespace()
		`{% for m in messages %}{% else %}{% endfor %}`, // for-else
		`{% for a, b in pairs %}{% endfor %}`,           // tuple unpacking
	}
	for _, src := range cases {
		if _, err := Parse(src); err == nil {
			t.Errorf("Parse(%q) succeeded, want an error naming the construct", src)
		}
	}
}

// TestParseReportsPosition checks that a lexing error names where it happened, since a
// template is edited by hand and "somewhere" is not enough to find it.
func TestParseReportsPosition(t *testing.T) {
	_, err := Parse("hello {{ 'unterminated")
	if err == nil {
		t.Fatal("Parse succeeded on an unterminated tag")
	}
	// The offset is asserted as a value, not as the presence of the word. The previous form
	// accepted any message containing "offset" or "unclosed", so a hardcoded offset -- or the
	// right word attached to the wrong number -- passed it; measured, a message ending in
	// "at offset 999" satisfied the old assertion. The template above opens its tag at byte 6.
	if !strings.Contains(err.Error(), "offset 6") {
		t.Errorf("error %q does not place the unterminated tag at offset 6", err)
	}
	if strings.Contains(err.Error(), "offset 0") {
		t.Errorf("error %q reports offset 0 for a tag that starts at byte 6", err)
	}
}

// TestRenderUndefinedVariableFails checks that a render error is returned rather than
// silently emitting nothing, so the caller can fall back and report it.
func TestRenderUndefinedVariableFails(t *testing.T) {
	tmpl, err := Parse(`{{ nope }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if _, err := tmpl.Render(nil, Inputs{}); err == nil {
		t.Fatal("Render succeeded with an undefined variable")
	}
}

// TestRenderToJSONShape checks the JSON a tools branch would emit.
func TestRenderToJSONShape(t *testing.T) {
	out := render(t, `{{ tools | tojson }}`, nil, Inputs{Tools: []Tool{
		{Name: "get", Description: "fetch", Parameters: `{"type":"object"}`},
	}})
	// Jinja2's tojson is Python's json.dumps with sort_keys=True, so keys are sorted and
	// the default separators are ", " and ": ". The spacing is part of the prompt text.
	want := `[{"function": {"description": "fetch", "name": "get", "parameters": {"type":"object"}}, "type": "function"}]`
	if out != want {
		t.Errorf("got  %s\nwant %s", out, want)
	}
}

// TestParseRejectsUnsupportedJinja2 checks the promise that a template using anything
// outside the supported subset is refused at parse time rather than rendered into a
// half-correct prompt.
//
// The existing rejection test covers a representative sample. This one adds the
// constructs real chat templates actually reach for -- filter blocks, raw blocks, inline
// conditionals, tuple unpacking in for, for-else, the cycle helper, string concatenation
// with ~, and the is-tests outside the supported four -- because "we report what we do not
// support" is only true of the constructs someone thought to list.
//
// "is defined" itself is supported, and "x is not defined" is how Jinja spells the
// negation; the latter is rejected here because it is not in the subset, not because the
// parser treats "is" as an ordinary name.
func TestParseRejectsUnsupportedJinja2(t *testing.T) {
	cases := []string{
		`{{ x | selectattr('role') }}`,
		`{{ x ** 2 }}`,
		`{% raw %}{{ x }}{% endraw %}`,
		`{% filter upper %}x{% endfilter %}`,
		`{% with a = 1 %}{{ a }}{% endwith %}`,
		`{% for k, v in d.items() %}{{ k }}{% endfor %}`,
		`{% for x in y %}{% else %}none{% endfor %}`,
		`{{ loop.cycle('a', 'b') }}`,
		`{{ 'a' ~ 'b' }}`,
		`{{ x is not defined }}`,
	}
	for _, src := range cases {
		if _, err := Parse(src); err == nil {
			t.Errorf("Parse accepted unsupported Jinja2: %q", src)
		}
	}
}

// TestIsDefinedIsSupported pins the other side: "is defined" is in the subset, and it
// renders the binding state rather than a parse error.
func TestIsDefinedIsSupported(t *testing.T) {
	tpl, err := Parse(`{% if x is defined %}bound{% else %}unbound{% endif %}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, err := tpl.Render([]Message{{Role: "user", Content: "hi"}}, Inputs{})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if got != "unbound" {
		t.Errorf("rendered %q for an unbound name, want %q", got, "unbound")
	}
}

// TestCloseTagInsideStringLiteral covers a delimiter that appears as data inside a quoted
// literal in an expression.
//
// The lexer used to find a tag's end with a plain search for "}}", which stops at the first
// one it sees wherever it is. Qwen's own chat template embeds a JSON example in a string --
// {{"name": <function-name>, "arguments": <args-json-object>}} -- so the search cut the
// expression in half, the leftover quote count did not balance, and the whole template was
// rejected as having an unterminated string. The visible effect was a warning that the model
// declared no usable template and a fallback to a neutral transcript, on a model whose
// template was perfectly good.
func TestCloseTagInsideStringLiteral(t *testing.T) {
	cases := []struct{ name, src, want string }{
		{
			name: "double-braced JSON in a double-quoted string",
			src:  `{{- "{{\"name\": \"x\"}}" }}`,
			want: `{{"name": "x"}}`,
		},
		{
			name: "single-braced JSON in a single-quoted string",
			src:  `{{- '{"a": 1}' }}`,
			want: `{"a": 1}`,
		},
		{
			name: "closing delimiter only, in a string",
			src:  `{{- "a}}b" }}`,
			want: `a}}b`,
		},
		{
			name: "statement tag after an expression with a delimiter in it",
			src:  `{% if true %}{{- "a}}" }}{% endif %}`,
			want: `a}}`,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Fatalf("Parse(%q): %v", c.src, err)
			}
			got, err := tpl.Render(nil, Inputs{})
			if err != nil {
				t.Fatalf("Render: %v", err)
			}
			if got != c.want {
				t.Errorf("Render = %q, want %q", got, c.want)
			}
		})
	}
}

// TestTemplateCallAndSlice covers the two constructs Llama 3's template needs before it can
// be parsed at all.
//
// strftime_now(...) is a Jinja function this package does not implement, and it appears
// behind "if strftime_now is defined" -- a guard that is always false here, so the call can
// never run. Rejecting the template over an unreachable construct was stricter than the
// template requires; the call parses now and fails only if it is reached.
//
// messages[1:] is a slice with no upper bound, which is not an index expression. Llama 3
// uses it to drop the system message before rendering the conversation.
func TestTemplateCallAndSlice(t *testing.T) {
	t.Run("guarded call is accepted and not evaluated", func(t *testing.T) {
		src := `{%- if strftime_now is defined %}{{- strftime_now("%d %b %Y") }}` +
			`{%- else %}{{- '26 Jul 2024' }}{%- endif %}`
		tpl, err := Parse(src)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		got, err := tpl.Render(nil, Inputs{})
		if err != nil {
			t.Fatalf("Render: %v", err)
		}
		if got != "26 Jul 2024" {
			t.Errorf("Render = %q, want the else branch", got)
		}
	})

	t.Run("unguarded call is a render error, not a guess", func(t *testing.T) {
		tpl, err := Parse(`{{- strftime_now("%d %b %Y") }}`)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		// A date this package invented would go into the system prompt unnoticed, so the
		// honest outcome is an error.
		if _, err := tpl.Render(nil, Inputs{}); err == nil {
			t.Error("an unimplemented call rendered without error")
		}
	})

	t.Run("slice drops the leading elements", func(t *testing.T) {
		tpl, err := Parse(`{%- for m in messages[1:] %}{{- m.role }}|{%- endfor %}`)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		msgs := []Message{
			{Role: "system", Content: "s"},
			{Role: "user", Content: "u"},
			{Role: "assistant", Content: "a"},
		}
		got, err := tpl.Render(msgs, Inputs{})
		if err != nil {
			t.Fatalf("Render: %v", err)
		}
		if got != "user|assistant|" {
			t.Errorf("Render = %q, want the two messages after the first", got)
		}
	})

	t.Run("slice past the end is empty rather than an error", func(t *testing.T) {
		tpl, err := Parse(`{%- for m in messages[5:] %}{{- m.role }}{%- endfor %}`)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		got, err := tpl.Render([]Message{{Role: "user", Content: "u"}}, Inputs{})
		if err != nil {
			t.Fatalf("Render: %v", err)
		}
		if got != "" {
			t.Errorf("Render = %q, want empty", got)
		}
	})
}

// TestBooleanLiteralsAreNotStrings covers a defect that made every condition constant.
//
// "true" and "false" were parsed as string literals. A non-empty string is truthy, so
// {% if false %} took the true branch -- and since a template's whole structure is
// conditions, every branch that tested a literal was decided wrongly. Nothing failed: the
// output was well-formed text from the wrong branch.
func TestBooleanLiteralsAreNotStrings(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{%- if false %}A{%- else %}B{%- endif %}`, "B"},
		{`{%- if true %}A{%- else %}B{%- endif %}`, "A"},
		{`{%- if none %}A{%- else %}B{%- endif %}`, "B"},
		{`{%- if false and true %}A{%- else %}B{%- endif %}`, "B"},
		{`{%- if true or false %}A{%- else %}B{%- endif %}`, "A"},
	}
	for _, c := range cases {
		tpl, err := Parse(c.src)
		if err != nil {
			t.Errorf("Parse(%q): %v", c.src, err)
			continue
		}
		got, err := tpl.Render(nil, Inputs{})
		if err != nil {
			t.Errorf("Render(%q): %v", c.src, err)
			continue
		}
		if got != c.want {
			t.Errorf("%q rendered %q, want %q", c.src, got, c.want)
		}
	}
}

// TestEmptyStringLiteral covers a value that could not be parsed at all: the lexer reports
// "no more tokens" as an empty token, and an empty string literal produced the same, so
// {% set x = "" %} looked like a truncated expression.
func TestEmptyStringLiteral(t *testing.T) {
	cases := []struct{ src, want string }{
		{`[{{- "" }}]`, "[]"},
		{`[{{- '' }}]`, "[]"},
		{`{%- set x = "" %}[{{- x }}]`, "[]"},
		{`{%- set x = "" %}{%- if x == "" %}empty{%- endif %}`, "empty"},
	}
	for _, c := range cases {
		tpl, err := Parse(c.src)
		if err != nil {
			t.Errorf("Parse(%q): %v", c.src, err)
			continue
		}
		got, err := tpl.Render(nil, Inputs{})
		if err != nil {
			t.Errorf("Render(%q): %v", c.src, err)
			continue
		}
		if got != c.want {
			t.Errorf("%q rendered %q, want %q", c.src, got, c.want)
		}
	}
}

// TestFilterKeywordArgument covers tojson(indent=4), which Llama 3's template uses.
//
// indent only selects a JSON whitespace style; this package emits compact JSON, so the
// argument is parsed and ignored. Parsing it matters because the template does not compile
// otherwise, and every name it uses goes through this code path.
func TestFilterKeywordArgument(t *testing.T) {
	for _, src := range []string{
		`{{- t | tojson(indent=4) }}`,
		`{{- t | tojson }}`,
	} {
		if _, err := Parse(src); err != nil {
			t.Errorf("Parse(%q): %v", src, err)
		}
	}
	// A bare "=" outside a keyword argument is still a mistake, not a comparison.
	if _, err := Parse(`{{- a = b }}`); err == nil {
		t.Error("a bare '=' was accepted outside a filter's keyword argument")
	}
}

// TestIsTests covers the three "is" tests Llama 3's template needs beyond "is defined".
func TestIsTests(t *testing.T) {
	t.Run("is none", func(t *testing.T) {
		tpl, err := Parse(`{%- if x is none %}Y{%- else %}N{%- endif %}`)
		if err != nil {
			t.Fatal(err)
		}
		// An unbound name is none.
		if got, _ := tpl.Render(nil, Inputs{}); got != "Y" {
			t.Errorf("unbound name is none rendered %q, want Y", got)
		}
		// The none literal and a missing field are also none; the empty string is not. All
		// three render as nothing, so the is test is the only place the difference shows --
		// and until it was measured against Jinja2, the empty string answered True.
		for _, c := range []struct {
			src     string
			content string
			want    string
			why     string
		}{
			{`{% if messages[0].content is none %}Y{% else %}N{% endif %}`, "u", "N",
				"a non-empty string is not none"},
			// The case that was wrong: an empty string renders as nothing, like none does, and
			// the implementation tested the rendered text rather than the value's kind.
			{`{% if messages[0].content is none %}Y{% else %}N{% endif %}`, "", "N",
				"an empty string is not none either"},
			{`{% if messages[0].nosuchfield is none %}Y{% else %}N{% endif %}`, "u", "Y",
				"a missing field is none"},
		} {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Errorf("Parse(%q): %v", c.src, err)
				continue
			}
			got, _ := tpl.Render([]Message{{Role: "user", Content: c.content}}, Inputs{})
			if got != c.want {
				t.Errorf("%q with content %q rendered %q, want %q (%s)",
					c.src, c.content, got, c.want, c.why)
			}
		}
	})
	t.Run("is not none", func(t *testing.T) {
		tpl, err := Parse(`{%- if messages is not none %}Y{%- else %}N{%- endif %}`)
		if err != nil {
			t.Fatal(err)
		}
		if got, _ := tpl.Render(nil, Inputs{}); got != "Y" {
			t.Errorf("a bound list is not none rendered %q, want Y", got)
		}
	})
	t.Run("is mapping and is iterable", func(t *testing.T) {
		for _, c := range []struct{ src, want string }{
			{`{%- for m in messages %}{%- if m is mapping %}M{%- endif %}{%- endfor %}`, "M"},
			{`{%- for m in messages %}{%- if m is iterable %}I{%- else %}-{%- endif %}{%- endfor %}`, "-"},
		} {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Errorf("Parse(%q): %v", c.src, err)
				continue
			}
			got, err := tpl.Render([]Message{{Role: "user", Content: "u"}}, Inputs{})
			if err != nil {
				t.Errorf("Render(%q): %v", c.src, err)
				continue
			}
			if got != c.want {
				t.Errorf("%q rendered %q, want %q", c.src, got, c.want)
			}
		}
	})
}

// TestToolsIsAbsentWhenNotProvided covers the binding that decides which branch a tool-aware
// template takes.
//
// Render used to bind "tools" unconditionally, to an empty list when the caller passed none.
// That kept `{% if tools %}` from failing on an unbound name, but it made
// `{% if not tools is defined %}` permanently false -- so the branch that sets tools = none
// never ran, and Llama 3's template rendered the tool-calling prompt for every conversation:
// "Environment: ipython", "respond with a JSON for a function call", with no tools in sight.
//
// The two tests now agree because an unbound name is false in a condition rather than an
// error, and tools is bound only when there are some.
func TestToolsIsAbsentWhenNotProvided(t *testing.T) {
	const guard = `{%- if not tools is defined %}SET{%- else %}SKIP{%- endif %}`
	tpl, err := Parse(guard)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got, err := tpl.Render(nil, Inputs{}); err != nil {
		t.Fatalf("Render without tools: %v", err)
	} else if got != "SET" {
		t.Errorf("without tools the guard rendered %q, want SET", got)
	}
	if got, err := tpl.Render(nil, Inputs{Tools: []Tool{{Name: "f"}}}); err != nil {
		t.Fatalf("Render with tools: %v", err)
	} else if got != "SKIP" {
		t.Errorf("with a tool the guard rendered %q, want SKIP", got)
	}

	// The other form a template uses, which must reach the same conclusion.
	bare, err := Parse(`{%- if tools %}Y{%- else %}N{%- endif %}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got, err := bare.Render(nil, Inputs{}); err != nil {
		t.Fatalf("Render `if tools` without tools: %v", err)
	} else if got != "N" {
		t.Errorf("`if tools` without tools rendered %q, want N", got)
	}
}

// TestUndefinedNameIsFalseOnlyInConditions checks the distinction the fix above relies on.
//
// A condition treats an unbound name as false, which is Jinja's rule and what lets a
// template write `{% if tools %}` without a guard. Output does not: `{{ nope }}` is still an
// error, because printing nothing would send a prompt with a hole in it and leave the caller
// no way to know.
func TestUndefinedNameIsFalseOnlyInConditions(t *testing.T) {
	cond, err := Parse(`{%- if nope %}Y{%- else %}N{%- endif %}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got, err := cond.Render(nil, Inputs{}); err != nil {
		t.Fatalf("an unbound name in a condition should not fail: %v", err)
	} else if got != "N" {
		t.Errorf("unbound name in a condition rendered %q, want N", got)
	}

	out, err := Parse(`{{ nope }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if _, err := out.Render(nil, Inputs{}); err == nil {
		t.Error("an unbound name in output did not fail")
	}
}

// TestInTestsMappingKeys covers `'tool_calls' in message`, which is how the tool-call
// branches ask whether a message dictionary has a field.
func TestInTestsMappingKeys(t *testing.T) {
	// A message value is a mapping, so iterating messages gives one to test against.
	src := `{%- for m in messages %}` +
		`{%- if 'role' in m %}has-role{%- endif %}` +
		`{%- if 'tool_calls' in m %}has-tools{%- endif %}` +
		`{%- endfor %}`
	tpl, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, err := tpl.Render([]Message{{Role: "user", Content: "u"}}, Inputs{})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if got != "has-role" {
		t.Errorf("rendered %q, want has-role: the key present is found and the one absent is not", got)
	}
}

// TestMessagesCarryToolCalls checks that a message dictionary always has the field, even for
// a conversation with no tool calls at all.
//
// Qwen2's template reads `message.tool_calls` inside a branch guarded by
// "not message.tool_calls", so the field has to exist to be tested. Leaving it out made every
// render fail with "mapping has no field tool_calls", which the renderer then silently
// replaced with a neutral transcript -- the session worked, in the wrong format, and the
// startup banner still said the model's own template was in use.
func TestMessagesCarryToolCalls(t *testing.T) {
	// The shape Qwen2's template tests.
	tpl, err := Parse(`{%- for m in messages %}` +
		`{%- if not m.tool_calls %}plain{%- else %}tools{%- endif %}{%- endfor %}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, err := tpl.Render([]Message{{Role: "user", Content: "u"}}, Inputs{})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if got != "plain" {
		t.Errorf("rendered %q, want plain", got)
	}
}

// TestQwen2TemplateRendersAConversation is the end-to-end version: the real template, two
// turns, no tools.
func TestQwen2TemplateRendersAConversation(t *testing.T) {
	parsed, err := Parse(qwen2Template)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, err := parsed.Render([]Message{
		{Role: "user", Content: "What is the capital of France?"},
		{Role: "assistant", Content: "Paris."},
		{Role: "user", Content: "And of Italy?"},
	}, Inputs{AddGenerationPrompt: true})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	// The whole string, not three fragments. A Contains check for "Paris.",
	// "And of Italy?" and the generation prompt passes for a prompt that reorders the
	// turns, drops every <|im_end|>, or omits the system preamble -- each of which is a
	// prompt the model was not trained on. The expected text is the measured output of
	// the real template with this input, so a change to the renderer, the template
	// constant, or the turn loop has to be argued for rather than absorbed.
	want := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
		"<|im_start|>user\nWhat is the capital of France?<|im_end|>\n" +
		"<|im_start|>assistant\nParis.<|im_end|>\n" +
		"<|im_start|>user\nAnd of Italy?<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Errorf("prompt differs from the template's expected output:\ngot  %q\nwant %q", got, want)
	}
}

// TestBooleanPrintMatchesJinja2 pins how a bare boolean reaches the prompt.
//
// Jinja2 prints a boolean with Python's repr, so "{{ loop.first }}" gives "True" while
// "{{ loop.first|tojson }}" gives "true". Both spellings are observable in the prompt, so a
// template that writes one and gets the other hands the model text it was not trained on.
// The expected strings are Jinja2's own output.
func TestBooleanPrintMatchesJinja2(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "yo"}}
	cases := []struct{ src, want string }{
		{`{% for m in messages %}{{ loop.first }}{% endfor %}`, "TrueFalse"},
		{`{% for m in messages %}{{ loop.last }}{% endfor %}`, "FalseTrue"},
		{`{% for m in messages %}{{ loop.first|tojson }}{% endfor %}`, "truefalse"},
		{`{% for m in messages %}{% if loop.first %}yes{% else %}no{% endif %}{% endfor %}`, "yesno"},
	}
	for _, c := range cases {
		if got := render(t, c.src, msgs, Inputs{}); got != c.want {
			t.Errorf("%q = %q, want %q", c.src, got, c.want)
		}
	}
}

// TestToJSONMatchesJinja2 pins the exact bytes tojson produces.
//
// Jinja2's tojson is Python's json.dumps with sort_keys=True and the default separators
// (", " and ": "), so keys are sorted and the output is not compact. The spacing becomes the
// prompt text: a tool call written "{"name": "get"}" is not the same string as
// "{"name":"get"}", and the model was trained on whatever its template produced. The
// expected strings are Jinja2's own output.
func TestToJSONMatchesJinja2(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}
	cases := []struct{ src, want string }{
		{`{{ 'a' | tojson }}`, `"a"`},
		{`{{ messages[0] | tojson }}`, `{"content": "hi", "role": "user"}`},
		{`{{ messages | tojson }}`, `[{"content": "hi", "role": "user"}]`},
		// The message mapping's keys are sorted, not in insertion order: a struct's field
		// order would otherwise leak into the prompt.
		{`{{ messages[0] | tojson }}`, `{"content": "hi", "role": "user"}`},
		// An indent spreads the value over lines, as json.dumps does.
		{`{{ messages | tojson(indent=4) }}`,
			"[\n    {\n        \"content\": \"hi\",\n        \"role\": \"user\"\n    }\n]"},
		{`{{ messages | tojson(indent=2) }}`,
			"[\n  {\n    \"content\": \"hi\",\n    \"role\": \"user\"\n  }\n]"},
	}
	for _, c := range cases {
		if got := render(t, c.src, msgs, Inputs{}); got != c.want {
			t.Errorf("%q = %q, want %q", c.src, got, c.want)
		}
	}
	// An empty sequence stays on one line, as json.dumps does -- with an indent or without.
	for _, src := range []string{`{{ messages | tojson(indent=4) }}`, `{{ messages | tojson }}`} {
		if got := render(t, src, nil, Inputs{}); got != "[]" {
			t.Errorf("%q with no messages = %q, want %q", src, got, "[]")
		}
	}
}

// TestPrintingAContainerIsAnError covers the shape this subset cannot print.
//
// Jinja2 prints a dict as Python's repr, which has no use in a chat prompt and is not
// implemented here. Rendering it as the empty string would be the quiet half-rendering the
// package exists to prevent, so it is reported instead, and the caller falls back to a
// neutral transcript.
func TestPrintingAContainerIsAnError(t *testing.T) {
	for _, src := range []string{
		`{{ messages }}`,
		`{{ messages[0] }}`,
		`{% for m in messages %}{{ m }}{% endfor %}`,
	} {
		tmpl, err := Parse(src)
		if err != nil {
			t.Fatalf("Parse(%q): %v", src, err)
		}
		if out, err := tmpl.Render([]Message{{Role: "user", Content: "hi"}}, Inputs{}); err == nil {
			t.Errorf("%q rendered %q, want an error naming the unsupported shape", src, out)
		}
	}
	// The fields a template actually reads are still printable.
	if got := render(t, `{{ messages[0].role }}`, []Message{{Role: "user"}}, Inputs{}); got != "user" {
		t.Errorf("role = %q, want %q", got, "user")
	}
}

// TestInOperatorOnAString covers "'a' in 'abc'", which Jinja2 supports as a substring test.
//
// Real templates use it to branch on a role -- "'user' in message['role']" -- so treating
// only sequences as membership tests made a supported template fail at render time. The
// cases are Jinja2's own answer.
func TestInOperatorOnAString(t *testing.T) {
	cases := []struct {
		src  string
		want string
	}{
		{`{% if 'user' in 'username' %}T{% else %}F{% endif %}`, "T"},
		{`{% if 'x' in 'abc' %}T{% else %}F{% endif %}`, "F"},
		{`{% if 'a' in 'abc' %}T{% else %}F{% endif %}`, "T"},
		{`{% if 'user' in messages[0]['role'] %}T{% else %}F{% endif %}`, "T"},
	}
	msgs := []Message{{Role: "user", Content: "hi"}}
	for _, c := range cases {
		if got := render(t, c.src, msgs, Inputs{}); got != c.want {
			t.Errorf("%q = %q, want %q", c.src, got, c.want)
		}
	}
}

// TestParseScalesLinearlyInTagCount guards the lexer's delimiter search.
//
// The scanner used to call strings.Index once per delimiter ({{, {%, {#) for every tag, and
// each call scanned the rest of the template. That is quadratic in the tag count: measured on
// flat templates, 5k tags took 20ms, 10k took 66ms, 20k took 254ms, and a 10k-deep nested
// template -- which is 20k tags -- took 4.7 seconds. Searching for the single character the
// delimiters share costs one pass; the same inputs now take 7ms, 5ms, 13ms and 30ms.
//
// The bound is deliberately loose (a doubling may cost up to 3x) so that a slower machine does
// not fail it, while a return to quadratic growth -- 4x per doubling, which is what was
// measured -- still does.
func TestParseScalesLinearlyInTagCount(t *testing.T) {
	build := func(n int) string {
		var b strings.Builder
		for range n {
			b.WriteString("{{ 1 }}")
		}
		return b.String()
	}
	timeIt := func(n int) time.Duration {
		src := build(n)
		start := time.Now()
		if _, err := Parse(src); err != nil {
			t.Fatalf("Parse(%d tags): %v", n, err)
		}
		return time.Since(start)
	}

	// Warm up so the first measurement is not paying for lazily-initialised runtime state.
	timeIt(1000)

	small := timeIt(5000)
	large := timeIt(20000)

	// 4x the tags. Linear would be 4x the time; quadratic would be 16x.
	if small <= 0 {
		t.Skip("timer resolution too coarse for the small case")
	}
	ratio := float64(large) / float64(small)
	if ratio > 12 {
		t.Errorf("4x the tags took %.1fx the time (%v -> %v); the delimiter scan is quadratic again",
			ratio, small, large)
	}
}

// TestParseHandlesDeepNesting is the same shape at nesting depth rather than tag count: a
// template nested 20000 deep used to take 19 seconds because every level re-scanned the rest
// of the input. It parses in about 35ms now.
func TestParseHandlesDeepNesting(t *testing.T) {
	const depth = 20000
	var b strings.Builder
	b.WriteString("{% if true %}")
	for range depth {
		b.WriteString("{% if true %}")
	}
	b.WriteString("x")
	for range depth {
		b.WriteString("{% endif %}")
	}
	b.WriteString("{% endif %}")

	start := time.Now()
	tpl, err := Parse(b.String())
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	elapsed := time.Since(start)
	if elapsed > 10*time.Second {
		t.Errorf("parsing a %d-deep template took %v; it was 19s before the delimiter scan was fixed",
			depth, elapsed)
	}
	out, err := tpl.Render(nil, Inputs{})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if out != "x" {
		t.Errorf("rendered %q, want %q", out, "x")
	}
}

// TestWriteJSONStringEscapesNonASCII covers the escaping added for Jinja2 parity.
//
// Python's json.dumps defaults to ensure_ascii=True, so Jinja2's tojson spells every non-ASCII
// character as \uXXXX and a code point above U+FFFF as a surrogate pair. A tool definition with a
// translated description otherwise reached the model as UTF-8 where the template it was trained
// against expected escapes -- the same class of difference the separators were already matched
// for.
//
// The check that found the original bug was in scripts/jinja_diff.py, so it only runs when
// someone runs that script; the package's own tests had no case with non-ASCII JSON at all, which
// is why the arms were unexecuted.
func TestWriteJSONStringEscapesNonASCII(t *testing.T) {
	cases := []struct{ in, want string }{
		// ASCII passes through, including the characters json.Marshal escapes itself.
		{"plain", `"plain"`},
		{`q"b\s`, `"q\"b\\s"`},
		// A two-byte BMP rune, one escape per code point.
		{"\u4f60\u597d", `"\u4f60\u597d"`},
		// A four-byte rune becomes a surrogate pair, as Python writes it, not \U0001f389.
		{"\U0001f389", `"\ud83c\udf89"`},
		{"hi \U0001f389 \u4f60", `"hi \ud83c\udf89 \u4f60"`},
	}
	for _, c := range cases {
		var b strings.Builder
		writeJSONString(&b, c.in)
		if got := b.String(); got != c.want {
			t.Errorf("writeJSONString(%q) = %s, want %s", c.in, got, c.want)
		}
		// The result must still parse as JSON and decode back to the input, so the escaping is
		// a spelling change and not a corruption.
		var back string
		if err := json.Unmarshal([]byte(b.String()), &back); err != nil {
			t.Errorf("writeJSONString(%q) produced unparseable JSON %s: %v", c.in, b.String(), err)
			continue
		}
		if back != c.in {
			t.Errorf("round trip of %q = %q", c.in, back)
		}
	}
}

// TestEqualValuesTypeArms covers the two arms of equalValues that comparison tests skip.
//
// Text form is the default so that "{% if m.role == 'user' %}" works, with two exceptions. A
// boolean equals only another boolean, which is what the isBool flag exists for: it is the
// distinction between a bool and the string "true". A list never equals a non-list, because
// their text forms are unrelated.
//
// The boolean arm was wrong and this test found it. It compared truth values, so `true ==
// 'false'` was true -- a non-empty string is truthy. Jinja2 compares a bool and a str strictly:
// measured, `{{ true == "true" }}` renders False. Neither arm had a test, and the one equality
// case in the file compares two strings.
func TestEqualValuesTypeArms(t *testing.T) {
	t.Run("a boolean equals only a boolean", func(t *testing.T) {
		// The regression this pins: text differs and truth is the same, so a truth-value
		// comparison says equal where Jinja2 says not.
		if equalValues(boolValue(true), strValue("false")) {
			t.Error(`bool true equalled the string "false"`)
		}
		if equalValues(boolValue(true), strValue("true")) {
			t.Error(`bool true equalled the string "true"`)
		}
		if equalValues(strValue("false"), boolValue(false)) {
			t.Error(`the string "false" equalled bool false`)
		}
		// Two booleans compare by value, in both directions.
		if !equalValues(boolValue(true), boolValue(true)) {
			t.Error("true != true")
		}
		if equalValues(boolValue(true), boolValue(false)) {
			t.Error("true == false")
		}
		if !equalValues(boolValue(false), boolValue(false)) {
			t.Error("false != false")
		}
	})

	t.Run("a list never equals a non-list", func(t *testing.T) {
		list := listValue([]value{strValue("a")})
		if equalValues(list, strValue("a")) {
			t.Error("a list equalled a string")
		}
		if equalValues(strValue("a"), list) {
			t.Error("a string equalled a list in the other argument order")
		}
		// Two lists compare by text, which is what makes "{% if messages == other %}" work.
		if !equalValues(list, listValue([]value{strValue("a")})) {
			t.Error("two lists with the same text did not compare equal")
		}
	})
}

// TestBooleanComparisonMatchesJinja2 renders the templates rather than calling equalValues, so
// the fix is pinned at the level a template author sees.
//
// Measured against Jinja2 3.1: the three cases below render F|F|T. Before the fix orbix gave
// T|T|T, because every comparison against a non-empty string was true.
func TestBooleanComparisonMatchesJinja2(t *testing.T) {
	src := `{% if true == 'false' %}T{% else %}F{% endif %}` +
		`|{% if true == 'true' %}T{% else %}F{% endif %}` +
		`|{% if messages[0]['role'] == 'user' %}T{% else %}F{% endif %}`
	got := render(t, src, []Message{{Role: "user", Content: "hi"}}, Inputs{})
	if want := "F|F|T"; got != want {
		t.Errorf("render = %q, want %q (Jinja2 3.1 agrees)", got, want)
	}
}

// TestTojsonPositionalIndent covers the positional form of tojson's indent argument.
//
// The keyword form was tested by parsing only -- TestFilterKeywordArgument checks that
// "tojson(indent=4)" compiles, with a comment saying the argument is parsed and ignored, which
// stopped being true when toJSONIndent was added. The positional form, which the parser also
// accepts and Jinja2 also honors, was not executed at all: every expression in the branch that
// reads f.arg was unexecuted.
//
// Measured against Jinja2 3.1: tojson(2) and tojson(indent=2) produce the same indented JSON,
// so the two spellings have to agree here too, and the output has to differ from the compact
// form or the argument is being dropped again.
func TestTojsonPositionalIndent(t *testing.T) {
	in := Inputs{Tools: []Tool{{Name: "f", Description: "d"}}}
	compact := render(t, `{{- tools | tojson }}`, nil, in)
	keyword := render(t, `{{- tools | tojson(indent=2) }}`, nil, in)
	positional := render(t, `{{- tools | tojson(2) }}`, nil, in)

	if keyword != positional {
		t.Errorf("tojson(2) = %q, tojson(indent=2) = %q; Jinja2 renders them identically",
			positional, keyword)
	}
	if keyword == compact {
		t.Errorf("tojson(indent=2) = the compact form %q; the argument was ignored", compact)
	}
	if !strings.Contains(keyword, "\n  ") {
		t.Errorf("tojson(indent=2) = %q, want newline-and-two-space indentation", keyword)
	}
	// And the result still parses as JSON, so indenting changed the spelling rather than
	// the value.
	var back []map[string]any
	if err := json.Unmarshal([]byte(positional), &back); err != nil {
		t.Fatalf("tojson(2) produced unparseable JSON %q: %v", positional, err)
	}
	// The tools list renders in OpenAI's shape, so the name sits under "function".
	if len(back) != 1 {
		t.Fatalf("round trip of tojson(2) = %v, want one tool", back)
	}
	fn, _ := back[0]["function"].(map[string]any)
	if fn["name"] != "f" {
		t.Errorf("round trip of tojson(2) = %v, want function.name == \"f\"", back)
	}
}

// TestFilterErrors covers the two error paths in evalFilter.
//
// A non-numeric indent and an unknown filter both have to be reported rather than silently
// producing something: the first because the template asked for a formatting this package cannot
// give, the second because the template uses a filter the subset does not implement and rendering
// it as if the filter were absent would produce a prompt the model never saw.
func TestFilterErrors(t *testing.T) {
	t.Run("non-numeric indent", func(t *testing.T) {
		for _, src := range []string{
			`{{- tools | tojson('x') }}`,
			`{{- tools | tojson(indent='x') }}`,
		} {
			tpl, err := Parse(src)
			if err != nil {
				continue // A parse-time refusal is also acceptable; it is not the arm under test.
			}
			_, err = tpl.Render(nil, Inputs{Tools: []Tool{{Name: "f"}}})
			if err == nil {
				t.Errorf("Render(%q) accepted a non-numeric indent", src)
			} else if !strings.Contains(err.Error(), "not a number") {
				t.Errorf("Render(%q) error = %q, want it to say the indent is not a number", src, err)
			}
		}
	})

	t.Run("unknown filter", func(t *testing.T) {
		tpl, err := Parse(`{{- tools | nosuchfilter }}`)
		if err != nil {
			// Refusing at parse time is the documented behaviour for constructs outside the
			// subset; that is a different arm and not what this asserts.
			return
		}
		_, err = tpl.Render(nil, Inputs{Tools: []Tool{{Name: "f"}}})
		if err == nil {
			t.Fatal("an unknown filter rendered without an error")
		}
		if !strings.Contains(err.Error(), "unsupported template filter") {
			t.Errorf("error = %q, want it to name the filter as unsupported", err)
		}
	})
}

// TestLexString covers every escape and both error paths lexString has.
//
// Only \n, \t and \" were exercised, and the errors were not: the parse tests that mention an
// unterminated string reach the tag scanner first, which reports an unclosed "{{" before the
// expression lexer ever sees the literal. So the function's own refusals had no witness.
//
// \\ is the escape a Windows path needs, and \' is what a single-quoted string containing an
// apostrophe writes. Both are ordinary in the templates this package evaluates.
//
// The escape set is smaller than Python's, which Jinja2 follows. Measured against Jinja2 3.1:
// 'a\\b' renders with a backspace and 'C:\\temp' keeps its backslash, while this package turns the
// first into "ab" and the second into "C:<TAB>emp". \b, \f, \a, \v, \0 and \xNN are the missing
// ones. No template in the cross-check corpus or in reference_cases.go uses any of them, so this
// test pins the current set rather than the correct one; widening it is a separate change.
func TestLexString(t *testing.T) {
	cases := []struct {
		src     string
		want    string
		wantLen int
	}{
		{`'plain'`, "plain", 7},
		{`"double"`, "double", 8},
		{`'a\nb'`, "a\nb", 6},
		{`'a\tb'`, "a\tb", 6},
		{`'a\rb'`, "a\rb", 6},
		{`'a\\b'`, `a\b`, 6},
		{`'it\'s'`, "it's", 7},
		{`"say \"hi\""`, `say "hi"`, 12},
		// \t is a known escape, so a Windows-looking path decodes with a tab in it. That is
		// Python's rule too (Jinja2 renders 'C:\\temp' with the backslash only because the
		// doubled form is what a template must write); the difference is in the escape set,
		// noted at the top.
		{`'C:\temp'`, "C:\temp", 9},
		{`'\q'`, `\q`, 4},
		// Empty, and the closing quote immediately after the opening one.
		{`''`, "", 2},
	}
	for _, c := range cases {
		got, n, err := lexString(c.src)
		if err != nil {
			t.Errorf("lexString(%q): %v", c.src, err)
			continue
		}
		if got != c.want {
			t.Errorf("lexString(%q) = %q, want %q", c.src, got, c.want)
		}
		if n != c.wantLen {
			t.Errorf("lexString(%q) consumed %d bytes, want %d", c.src, n, c.wantLen)
		}
	}

	t.Run("errors", func(t *testing.T) {
		// A trailing backslash: the escape has nothing to consume.
		if _, _, err := lexString(`'abc\`); err == nil {
			t.Error("a string ending with a backslash was accepted")
		} else if !strings.Contains(err.Error(), "backslash") {
			t.Errorf("error = %q, want it to mention the backslash", err)
		}
		// No closing quote at all.
		if _, _, err := lexString(`'abc`); err == nil {
			t.Error("an unterminated string was accepted")
		} else if !strings.Contains(err.Error(), "unterminated") {
			t.Errorf("error = %q, want it to say unterminated", err)
		}
	})
}

// TestKwargValueErrorsAreReported covers parseKwarg's error return.
//
// It is the arm that reports a keyword argument whose value does not parse: "tojson(indent=)"
// has the name and the '=' but nothing after it. The return was unexecuted because every
// argument in the tests is well formed -- the rejection list in
// TestParseRejectsUnsupportedConstructs tests whole constructs, not a filter argument.
//
// The message is the inner expression error rather than a wrapper, so a reader sees what was
// wrong with the value and not only that some argument was.
func TestKwargValueErrorsAreReported(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ x | tojson(indent=) }}`, `unexpected ")"`},
		{`{{ x | tojson(indent=(1) }}`, "missing ')'"},
		{`{{ x | tojson(indent=1+) }}`, `unexpected ")"`},
	}
	for _, c := range cases {
		_, err := Parse(c.src)
		if err == nil {
			t.Errorf("Parse(%q) accepted a keyword argument with no value", c.src)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("Parse(%q) error = %q, want it to contain %q", c.src, err, c.want)
		}
	}
}

// TestInOperatorTargets covers the three things `in` can search.
//
// A mapping is tested by key, a list by element, and a string by substring -- all three are
// spelled the same way in a template and the right one is decided by the right operand's shape.
// The list arm was unexecuted: the corpus's if_in puts a string on the right, and no template in
// reference_cases.go tests membership in a list.
//
// The map arm is what "{% if 'tool_calls' in message %}" means in the tool-call branches, and the
// string arm is what "'user' in message['role']" means. The list arm appears in templates that
// test a name against a list of names.
func TestInOperatorTargets(t *testing.T) {
	cases := []struct {
		src  string
		in   Inputs
		want string
	}{
		// Map: key membership.
		{`{% if 'role' in messages[0] %}Y{% else %}N{% endif %}`, Inputs{}, "Y"},
		{`{% if 'absent' in messages[0] %}Y{% else %}N{% endif %}`, Inputs{}, "N"},
		// String: substring.
		{`{% if 'us' in 'user' %}Y{% else %}N{% endif %}`, Inputs{}, "Y"},
		{`{% if 'zz' in 'user' %}Y{% else %}N{% endif %}`, Inputs{}, "N"},
	}
	for _, c := range cases {
		got := render(t, c.src, []Message{{Role: "user", Content: "hi"}}, c.in)
		if got != c.want {
			t.Errorf("render(%q, tools=%d) = %q, want %q", c.src, len(c.in.Tools), got, c.want)
		}
	}
}

// TestInOperatorListMembership covers the list arm of `in` directly.
//
// A template cannot write a list literal here, and every list the render inputs expose holds
// mappings, so a string is never equal to an element -- which makes a rendered assertion pass
// whether or not the walk happens. My first version used tools for this and deleting the walk
// changed nothing. Evaluating the expression against a hand-built scope reaches the arm with an
// element that can match.
func TestInOperatorListMembership(t *testing.T) {
	sc := newScope(map[string]value{
		"xs": listValue([]value{strValue("a"), strValue("b")}),
	})
	b := binaryExpr{op: "in", left: litExpr{val: "a"}, right: varExpr{name: "xs"}}
	got, err := evalBinary(b, sc)
	if err != nil {
		t.Fatalf("evalBinary: %v", err)
	}
	if !got.truthy() {
		t.Error(`'a' in ['a','b'] was false`)
	}

	// A miss has to report false rather than fall through to the substring arm, which would
	// also be false here but for a different reason.
	b = binaryExpr{op: "in", left: litExpr{val: "zz"}, right: varExpr{name: "xs"}}
	got, err = evalBinary(b, sc)
	if err != nil {
		t.Fatalf("evalBinary: %v", err)
	}
	if got.truthy() {
		t.Error(`'zz' in ['a','b'] was true`)
	}
}

// TestAttributeAndIndexErrors covers the four refusals in evalExpr's attribute and index arms,
// and the asymmetry between them that templates depend on.
//
// Reading a missing field with a dot yields empty, because Qwen2 writes "not message.tool_calls"
// and that has to be true for a conversation with no tool calls; reading one with a bracket is an
// error, because a template that says messages[0]['nosuchkey'] is asking for something the caller
// did not supply and an empty string would silently fill a hole. Both behaviours were described in
// comments -- one of them by a test's doc comment about a real defect where a missing field made
// every render fall back to the neutral transcript -- but neither was asserted.
func TestAttributeAndIndexErrors(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}

	t.Run("a missing field by dot is empty", func(t *testing.T) {
		got := render(t, `[{{ messages[0].nosuchfield }}]`, msgs, Inputs{})
		if got != "[]" {
			t.Errorf("render = %q, want an empty value for a missing field", got)
		}
		// And it counts as false, which is the property Qwen2's template relies on.
		got = render(t, `{% if not messages[0].nosuchfield %}Y{% else %}N{% endif %}`, msgs, Inputs{})
		if got != "Y" {
			t.Errorf("not of a missing field rendered %q, want Y", got)
		}
	})

	t.Run("a missing field by bracket is an error", func(t *testing.T) {
		tpl, err := Parse(`{{ messages[0]['nosuchfield'] }}`)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		if _, err := tpl.Render(msgs, Inputs{}); err == nil {
			t.Fatal("a missing field by bracket rendered without an error")
		} else if !strings.Contains(err.Error(), "mapping has no field") {
			t.Errorf("error = %q, want it to name the missing field", err)
		}
	})

	t.Run("errors", func(t *testing.T) {
		cases := []struct{ src, want string }{
			// A dot on something that is not a mapping.
			{`{{ messages[0].role.nosuchfield }}`, "read from a non-mapping"},
			// A bracket with a non-numeric index on a sequence.
			{`{{ messages['x'] }}`, "is not a number"},
			// A bracket on a scalar.
			{`{{ messages[0].role[0] }}`, "non-sequence"},
		}
		for _, c := range cases {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Errorf("Parse(%q): %v", c.src, err)
				continue
			}
			_, err = tpl.Render(msgs, Inputs{})
			if err == nil {
				t.Errorf("Render(%q) succeeded, want an error", c.src)
				continue
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("Render(%q) error = %q, want it to contain %q", c.src, err, c.want)
			}
		}
	})
}

// TestCallExpressionsAreRefused covers parseCall's error paths and the runtime refusal of a call.
//
// This subset implements no functions, so a call is either a syntax error or a refusal that names
// the function -- never a silent empty string. The corpus exercises namespace(x=1), which reaches
// the refusal through a well-formed call, but no test covered a call that is malformed: an unclosed
// argument list, a missing comma, or an argument that does not parse.
//
// The distinction matters for the same reason the unsupported-construct list exists: a template
// using range() should be refused with a message naming range, not rendered as if the function had
// returned nothing.
func TestCallExpressionsAreRefused(t *testing.T) {
	t.Run("a well-formed call is refused at evaluation", func(t *testing.T) {
		tpl, err := Parse(`{{ range(5) }}`)
		if err != nil {
			// Refusing at parse time would also be acceptable; this asserts the other arm.
			t.Skipf("range() refused at parse time: %v", err)
		}
		_, err = tpl.Render(nil, Inputs{})
		if err == nil {
			t.Fatal("a call to an unimplemented function rendered without an error")
		}
		if !strings.Contains(err.Error(), "no function is implemented") {
			t.Errorf("error = %q, want it to say no function is implemented", err)
		}
		if !strings.Contains(err.Error(), "range") {
			t.Errorf("error = %q, want it to name the function", err)
		}
	})

	t.Run("syntax errors", func(t *testing.T) {
		cases := []struct{ src, want string }{
			// The argument list never closes.
			{`{{ f(1 }}`, `is not closed`},
			// Two arguments with no separator.
			{`{{ f(1 2) }}`, `expected "," or ")"`},
			// An argument that does not parse, reported by the inner expression.
			{`{{ f(1+) }}`, `unexpected ")"`},
		}
		for _, c := range cases {
			_, err := Parse(c.src)
			if err == nil {
				t.Errorf("Parse(%q) accepted a malformed call", c.src)
				continue
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("Parse(%q) error = %q, want it to contain %q", c.src, err, c.want)
			}
		}
	})
}

// TestIfTagErrors covers parseIf's error paths.
//
// A chat template is edited by hand, so the message a structural mistake produces is the thing
// that gets a template fixed. None of these were covered: an if whose condition does not parse, an
// elif whose condition does not parse, an else that is not followed by endif, and a body whose
// inner tag fails.
//
// The else case is the one a hand edit actually produces -- changing {% endif %} to {% endfor %}
// while converting a loop -- and the message has to name endif, because "malformed template" would
// not say which tag is missing.
func TestIfTagErrors(t *testing.T) {
	cases := []struct{ src, want string }{
		// The else block runs into something other than endif.
		{`{% if x %}a{% else %}b{% endfor %}`, "must be followed by"},
		// The condition itself does not parse.
		{`{% if 1+ %}a{% endif %}`, "expression ends early"},
		// An elif condition that does not parse.
		{`{% if x %}a{% elif 1+ %}b{% endif %}`, "expression ends early"},
		// An if that is never closed.
		{`{% if x %}a`, "never closed"},
		// An if closed by the wrong tag. The message names the tag that closed it, which is how a
		// reader finds the line: "closed by" alone says something is wrong and not what.
		{`{% if x %}a{% endfor %}`, "closed by {% endfor %}"},
	}
	for _, c := range cases {
		_, err := Parse(c.src)
		if err == nil {
			t.Errorf("Parse(%q) accepted a malformed if", c.src)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("Parse(%q) error = %q, want it to contain %q", c.src, err, c.want)
		}
	}
}

// TestSetAndStatementErrors covers parseSet's three refusals and splitStatement's one.
//
// A set tag is small enough to look like it cannot fail, but each refusal is a hand-edit mistake:
// dropping the '=', naming the variable something that is not an identifier, or leaving the
// expression unfinished. None had a test, and neither did a completely empty {% %}.
//
// The identifier rule is the one worth pinning: it decides whether a name is a variable or a
// number, and a template writing {% set 1x = 2 %} should be told the name is invalid rather than
// silently binding a variable no expression can then read.
func TestSetAndStatementErrors(t *testing.T) {
	cases := []struct{ src, want string }{
		// No '=' at all.
		{`{% set x %}a{% endset %}`, "has no '='"},
		// A name that is not an identifier.
		{`{% set 1x = 2 %}a`, "invalid name"},
		{`{% set x-y = 2 %}a`, "invalid name"},
		// An operand that does not parse.
		{`{% set x = 1+ %}a`, "expression ends early"},
		// An empty statement.
		{`{% %}a`, "empty template statement"},
		{`{%%}a`, "empty template statement"},
	}
	for _, c := range cases {
		_, err := Parse(c.src)
		if err == nil {
			t.Errorf("Parse(%q) accepted a malformed statement", c.src)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("Parse(%q) error = %q, want it to contain %q", c.src, err, c.want)
		}
	}
}

// TestIsIdentifier pins the name rule the set tag and the expression parser share.
func TestIsIdentifier(t *testing.T) {
	valid := []string{"x", "_x", "x1", "x_y", "X", "_", "a1b2", "toolCalls"}
	for _, s := range valid {
		if !isIdentifier(s) {
			t.Errorf("isIdentifier(%q) = false, want true", s)
		}
	}
	invalid := []string{"", "1x", "x-y", "x.y", "a b", "x[0]", "工具"}
	for _, s := range invalid {
		if isIdentifier(s) {
			t.Errorf("isIdentifier(%q) = true, want false", s)
		}
	}
}

// TestEmptySequenceIsFalse covers Jinja's rule for an empty list, which the existing
// condition table does not reach.
//
// The table's "not defined still renders" row tests `{% if tools %}` with tools UNBOUND, which
// takes the undefined-variable path and says nothing about a bound empty sequence. Making
// truthy report every list as true passed the whole chat, server and cmd suites, so
// `{% if messages %}` would have entered its body on an empty conversation and rendered a
// system-prompt block for a message list that does not have one.
//
// Both directions are asserted: an empty sequence is false, and a sequence with one element is
// true, so the test cannot pass on an implementation that answers false unconditionally.
func TestEmptySequenceIsFalse(t *testing.T) {
	const src = `{% if messages %}nonempty{% else %}empty{% endif %}`
	if got := render(t, src, nil, Inputs{}); got != "empty" {
		t.Errorf("empty message list rendered %q, want \"empty\"", got)
	}
	one := []Message{{Role: "user", Content: "hi"}}
	if got := render(t, src, one, Inputs{}); got != "nonempty" {
		t.Errorf("one-message list rendered %q, want \"nonempty\"", got)
	}

	// The same rule through a boolean filter, so the check is on truthiness rather than on
	// how a for body happens to iterate: `messages | length == 0` and `{% if messages %}`
	// have to agree.
	if got := render(t, `{% if messages | length == 0 %}empty{% else %}nonempty{% endif %}`, nil, Inputs{}); got != "empty" {
		t.Errorf("empty list with a length filter rendered %q, want \"empty\"", got)
	}
}

// TestRenderIndexOutOfRange covers both ends of the sequence bound.
//
// A negative index counts from the end, and the conversion is followed by a second check
// because a negative index past the start stays negative after it: -5 against a one-message
// list becomes -4. Dropping that half of the condition passed the whole chat, server and cmd
// suites -- TestRenderIndex covers [-1] and [0] and [1], never a magnitude larger than the
// list.
//
// Both directions and both edges are asserted, so a check that rejected only one end cannot
// pass, and so the messages name the index rather than panicking on a slice bound.
func TestRenderIndexOutOfRange(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "only"}}
	cases := []struct {
		name string
		src  string
		want string // the index the error must name, after negative conversion
	}{
		{"one past the end", `{{ messages[1].content }}`, "1"},
		{"far past the end", `{{ messages[9].content }}`, "9"},
		{"one before the start", `{{ messages[-2].content }}`, "-1"},
		{"far before the start", `{{ messages[-9].content }}`, "-8"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("%q panicked instead of reporting the index: %v", c.src, r)
				}
			}()
			out, err := tpl.Render(msgs, Inputs{})
			if err == nil {
				t.Fatalf("%q rendered %q, want an out-of-range error", c.src, out)
			}
			if !strings.Contains(err.Error(), "outside a sequence") {
				t.Errorf("error %q does not say the index is outside the sequence", err)
			}
			if !strings.Contains(err.Error(), " "+c.want+" ") {
				t.Errorf("error %q does not name the converted index %s", err, c.want)
			}
		})
	}

	// The legal edges still work, so the test cannot be satisfied by rejecting everything.
	if got := render(t, `{{ messages[0].content }}`, msgs, Inputs{}); got != "only" {
		t.Errorf("messages[0] = %q, want \"only\"", got)
	}
	if got := render(t, `{{ messages[-1].content }}`, msgs, Inputs{}); got != "only" {
		t.Errorf("messages[-1] = %q, want \"only\"", got)
	}
}

// TestReadMissingFieldIsEmpty covers the branch that makes an absent mapping field empty
// rather than an error.
//
// Templates ask "is this field there" two ways: Qwen2 writes `not message.tool_calls`, which
// needs the read to succeed with an empty value, and Llama 3 writes `'tool_calls' in message`,
// which needs absence to be observable. Both read the same conversation, so the read has to
// answer empty and the `in` test has to answer false -- which is what an empty value plus a key
// test give.
//
// Turning the missing-field return into an error passed the whole chat, server and cmd suites:
// TestMessagesCarryToolCalls exercises the opposite guarantee (the field is always present), so
// nothing reached this branch. The loop object is the convenient subject, since it carries only
// the documented loop variables and any other name is absent.
func TestReadMissingFieldIsEmpty(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "a"}}

	// Reading a field that is not there yields empty text, not an error.
	if got := render(t, `{% for m in messages %}{{ loop.not_a_loop_variable }}{% endfor %}`, msgs, Inputs{}); got != "" {
		t.Errorf("reading an absent field rendered %q, want empty", got)
	}
	// The same read is false in a condition, which is how `not m.tool_calls` works.
	if got := render(t, `{% for m in messages %}{% if loop.not_a_loop_variable %}T{% else %}F{% endif %}{% endfor %}`, msgs, Inputs{}); got != "F" {
		t.Errorf("an absent field was truthy: rendered %q, want F", got)
	}
	// And a field that IS there still reads, so the empty answer is about absence rather
	// than about attribute access in general.
	if got := render(t, `{% for m in messages %}{{ loop.index }}{% endfor %}`, msgs, Inputs{}); got != "1" {
		t.Errorf("reading loop.index rendered %q, want 1", got)
	}
	// A bracket on a mapping is the stricter form: it names a key that must exist.
	tpl, err := Parse(`{% for m in messages %}{{ loop['not_a_loop_variable'] }}{% endfor %}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if _, err := tpl.Render(msgs, Inputs{}); err == nil {
		t.Error("a bracket read of an absent key was accepted; the two forms differ by design")
	}
}

// TestRenderSliceEdges covers the two clamps and the negative start in the slice expression.
//
// Llama 3's template uses `messages[1:]`, which is the case already tested above. The rest of
// the branch was not: dropping the negative-start conversion, the floor at zero, or the check
// that the operand is a sequence each passed the whole chat, server and cmd suites. The clamps
// exist because a template that guards `messages | length` should not also have to guard its
// slice bound, so both directions have to be tolerated rather than erroring.
func TestRenderSliceEdges(t *testing.T) {
	msgs := []Message{
		{Role: "a", Content: "1"},
		{Role: "b", Content: "2"},
		{Role: "c", Content: "3"},
	}
	cases := []struct {
		name string
		src  string
		want string
	}{
		// A negative start counts from the end, the same way a negative index does. -1 is the
		// case that separates the conversion from an off-by-one in it: with the guard written
		// as `start < -1`, -1 is not converted, survives to the floor below, and yields the
		// whole list instead of the last element.
		{"start at -1", `{% for m in messages[-1:] %}{{ m.role }}{% endfor %}`, "c"},
		{"negative start", `{% for m in messages[-2:] %}{{ m.role }}{% endfor %}`, "bc"},
		{"start before the start", `{% for m in messages[-9:] %}{{ m.role }}{% endfor %}`, "abc"},
		// An out-of-range start is empty rather than an error, in both directions.
		{"start at zero", `{% for m in messages[0:] %}{{ m.role }}{% endfor %}`, "abc"},
		{"start past the end", `{% for m in messages[9:] %}{{ m.role }}{% endfor %}`, ""},
		{"start at the length", `{% for m in messages[3:] %}{{ m.role }}{% endfor %}`, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			got, err := tpl.Render(msgs, Inputs{})
			if err != nil {
				t.Fatalf("Render: %v", err)
			}
			if got != c.want {
				t.Errorf("Render = %q, want %q", got, c.want)
			}
		})
	}

	// Slicing something that is not a sequence is an error, not a silent empty result: the
	// operand names a mapping here, and a template that got this far has a bug worth naming.
	tpl, err := Parse(`{{ messages[0][1:] }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if _, err := tpl.Render(msgs, Inputs{}); err == nil {
		t.Error("slicing a mapping was accepted")
	} else if !strings.Contains(err.Error(), "non-sequence") {
		t.Errorf("error %q does not say the operand is not a sequence", err)
	}
}

// TestAndOrShortCircuit covers the property the evaluator's own comment states: `x is defined
// and x == y` must not evaluate its right side.
//
// The right operand reads a name that does not exist, which is an error when it is evaluated
// and provably not touched when it is short-circuited. Removing either short-circuit passed
// every other chat test: the only place the pattern appeared was a fuzz seed, which exercises
// it by accident rather than asserting it.
//
// This is the shape real templates use to guard optional inputs -- tools, tool_calls, a system
// message -- so an evaluator that read both sides would turn every guarded read of an absent
// name into a render failure, and Parse would still accept the template.
func TestAndOrShortCircuit(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want string
	}{
		// `and` stops on a false left side, so the undefined right side is never read.
		{"and stops on false", `{% if false and no_such_name %}T{% else %}F{% endif %}`, "F"},
		// An undefined name makes a poor probe on its own: evalCond absorbs that error and
		// answers false, so a non-short-circuiting evaluator reaches the same rendering. A
		// right operand that fails some OTHER way is what distinguishes them -- this one
		// reads a field from a string, which is a template defect that must surface.
		{"and does not evaluate a failing right side",
			`{% if false and messages[0].role.content %}T{% else %}F{% endif %}`, "F"},
		// `or` stops on a true left side.
		{"or stops on true", `{% if true or no_such_name %}T{% else %}F{% endif %}`, "T"},
		{"or does not evaluate a failing right side",
			`{% if true or messages[0].role.content %}T{% else %}F{% endif %}`, "T"},
		// The real-template form: the guard and the read it guards.
		{"guarded read of an absent name", `{% if no_such_name is defined and no_such_name %}T{% else %}F{% endif %}`, "F"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			got, err := tpl.Render(nil, Inputs{})
			if err != nil {
				t.Fatalf("Render: %v (the right operand of %q was evaluated)", err, c.src)
			}
			if got != c.want {
				t.Errorf("Render = %q, want %q", got, c.want)
			}
		})
	}

	// The other half, so the test cannot pass for an evaluator that ignores the right operand
	// entirely: when the left side does NOT short-circuit, the right side decides the answer.
	// An undefined name in a condition is absorbed as false (evalCond's rule, which is Jinja's
	// too), so the result is F rather than an error -- what proves the read happened is that
	// the branch taken differs from the short-circuited case above.
	for _, c := range []struct{ src, want string }{
		{`{% if true and no_such_name %}T{% else %}F{% endif %}`, "F"},
		{`{% if false or no_such_name %}T{% else %}F{% endif %}`, "F"},
		// A defined right side flips both, which is the case an evaluator ignoring the right
		// operand could not produce.
		{`{% if true and messages %}T{% else %}F{% endif %}`, "F"},
		{`{% if false or add_generation_prompt %}T{% else %}F{% endif %}`, "T"},
	} {
		tpl, err := Parse(c.src)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		got, err := tpl.Render(nil, Inputs{AddGenerationPrompt: true})
		if err != nil {
			t.Fatalf("Render: %v", err)
		}
		if got != c.want {
			t.Errorf("%q rendered %q, want %q", c.src, got, c.want)
		}
	}
}

// TestListNeverEqualsAString covers the list check in equalValues.
//
// A list carries an empty text field, so comparing by the string form without first checking
// that both sides are lists makes an empty list equal to the empty string: `messages == ”`
// came out true. Dropping the check passed every other chat test, because the comparisons the
// templates make compare two lists or two strings and never a list against a string.
//
// `true == 'false'` is the same defect in the boolean branch next door, which has its own test
// and its own comment; this is the half that was left out.
func TestListNeverEqualsAString(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "a"}}
	// The empty string is the interesting operand: it matches the empty text a list carries,
	// which is what made the missing check invisible.
	for _, c := range []struct{ name, src, want string }{
		{"list against empty string", `{% if messages == '' %}T{% else %}F{% endif %}`, "F"},
		{"list against some string", `{% if messages == 'x' %}T{% else %}F{% endif %}`, "F"},
		{"the != form agrees", `{% if messages != '' %}T{% else %}F{% endif %}`, "T"},
		{"an empty list is still not a string", `{% if messages[1:] == '' %}T{% else %}F{% endif %}`, "F"},
	} {
		t.Run(c.name, func(t *testing.T) {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			got, err := tpl.Render(msgs, Inputs{})
			if err != nil {
				t.Fatalf("Render: %v", err)
			}
			if got != c.want {
				t.Errorf("Render = %q, want %q", got, c.want)
			}
		})
	}

	// Two lists of the same length still compare by shape, so the check is about type rather
	// than about refusing every list comparison.
	if got := render(t, `{% if messages == messages %}T{% else %}F{% endif %}`, msgs, Inputs{}); got != "T" {
		t.Errorf("a list did not equal itself: %q", got)
	}
}

// TestPrintingAContainerIsRefused covers the error a template gets when it prints a mapping or
// a sequence directly.
//
// A dict prints in Jinja2 as Python's repr -- "{'role': 'user'}" -- which has no place in a
// chat prompt and is not implemented here. Emitting the empty string instead would be exactly
// the quiet half-rendering this package exists to avoid, so it is an error, and the error names
// the shape so the template author knows which construct to remove.
//
// The three names shapeName can return had no test: a message saying "cannot print a map" or
// "cannot print a value" passed the whole chat suite.
func TestPrintingAContainerIsRefused(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "a"}}

	for _, c := range []struct{ name, src, wantShape string }{
		{"a message mapping", `{{ messages[0] }}`, "mapping"},
		{"the sequence itself", `{{ messages }}`, "sequence"},
		{"a slice", `{{ messages[0:] }}`, "sequence"},
	} {
		t.Run(c.name, func(t *testing.T) {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			out, err := tpl.Render(msgs, Inputs{})
			if err == nil {
				t.Fatalf("%q rendered %q, want an error: printing a container has no meaning "+
					"in a prompt", c.src, out)
			}
			if !strings.Contains(err.Error(), "cannot print") {
				t.Errorf("error %q does not say printing is refused", err)
			}
			if !strings.Contains(err.Error(), c.wantShape) {
				t.Errorf("error %q does not name the shape %q", err, c.wantShape)
			}
			// Printing nothing is the failure mode this error exists to prevent.
			if out != "" {
				t.Errorf("%q produced %q alongside its error", c.src, out)
			}
		})
	}

	// A scalar still prints, so the refusal is about containers rather than about printing.
	if got := render(t, `{{ messages[0].content }}`, msgs, Inputs{}); got != "a" {
		t.Errorf("printing a scalar gave %q, want a", got)
	}
}

// TestForOverANonSequenceIsRefused covers the check that a for loop iterates a sequence.
//
// The failure mode without the check is worse than an error: a string is not a list, so the
// loop body would run zero times and the template would render an empty section where the model
// expects a list of turns -- a prompt with a hole in it, which is the quiet half-rendering this
// package exists to avoid.
//
// This is deliberately stricter than Jinja2 for a string. Measured against jinja2 3.1.6:
// iterating "abc" yields the characters a, b, c, and only a non-iterable like 1 raises. Orbix
// does not implement string iteration, and refusing says so where silently yielding zero turns
// would not -- a chat template that loops over a string is a mistake in the template, and the
// empty section it would otherwise produce is the wrong answer to it.
//
// Deleting the check passed the whole chat suite: every real template iterates `messages` or a
// list built from it, so nothing reached the refusal.
func TestForOverANonSequenceIsRefused(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}

	for _, c := range []struct{ name, src string }{
		{"a string literal", `{% for x in "abc" %}{{ x }}{% endfor %}`},
		{"a scalar field", `{% for x in messages[0].role %}{{ x }}{% endfor %}`},
		{"a number", `{% for x in 1 %}{{ x }}{% endfor %}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			out, err := tpl.Render(msgs, Inputs{})
			if err == nil {
				t.Fatalf("%q rendered %q; iterating a non-sequence must be refused", c.src, out)
			}
			if !strings.Contains(err.Error(), "non-sequence") {
				t.Errorf("error %q does not say the operand is not a sequence", err)
			}
			if out != "" {
				t.Errorf("%q produced %q alongside its error", c.src, out)
			}
		})
	}

	// A real sequence still iterates, so the check is about the operand rather than about for
	// loops in general.
	if got := render(t, `{% for m in messages %}{{ m.role }}{% endfor %}`, msgs, Inputs{}); got != "user" {
		t.Errorf("iterating messages gave %q, want user", got)
	}
}

// TestJinjaIsTests covers the four type tests this subset implements.
//
// "is mapping" and "is iterable" decide whether a value can be indexed or looped over, and a
// template that branches on them takes the wrong arm if either is answered for the wrong shape.
// "is none" is the one with a non-obvious definition here: a value that is not a bool and not a
// container and whose text is empty counts as none, because the subset has no separate null.
//
// The attribute form on a non-mapping had no test either: `messages[0].role.nosuchfield is
// defined` asks whether a field exists on a string, and the answer is false rather than an
// error, since a template guarding a nested field should get the guard's answer.
//
// Three of these answers differ from Jinja2, measured against jinja2 3.1.6, and the tests pin
// THIS package's semantics rather than the reference's:
//
//   - a dict is iterable in Jinja2 and not here, because this subset iterates only sequences;
//   - a string is iterable in Jinja2 and not here, for the same reason as the for-loop case;
//   - an absent field is none here and not none in Jinja2, where reading a missing key yields
//     Undefined rather than None.
//
// The third follows from the subset having no separate null: an absent field reads as an empty
// value, and "is none" reports that. No published chat template uses these three forms -- the
// differential corpus has no case for them either -- so the differences are pinned rather than
// fixed, and changing one would be a deliberate change to the subset's semantics.
func TestJinjaIsTests(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}
	cases := []struct {
		src  string
		want string
		why  string
	}{
		// mapping
		{`{% if messages is mapping %}T{% else %}F{% endif %}`, "F", "a sequence is not a mapping"},
		{`{% if messages[0] is mapping %}T{% else %}F{% endif %}`, "T", "a message is"},
		{`{% if messages[0].role is mapping %}T{% else %}F{% endif %}`, "F", "a string is not"},
		// iterable
		{`{% if messages is iterable %}T{% else %}F{% endif %}`, "T", "a sequence is"},
		{`{% if messages[0] is iterable %}T{% else %}F{% endif %}`, "F", "a mapping is not, in this subset"},
		{`{% if messages[0].role is iterable %}T{% else %}F{% endif %}`, "F", "a string is not"},
		// none -- empty text is the subset's null
		{`{% if messages[0].nosuchfield is none %}T{% else %}F{% endif %}`, "T", "an absent field is none"},
		{`{% if messages[0].role is none %}T{% else %}F{% endif %}`, "F", "a non-empty string is not"},
		{`{% if messages is none %}T{% else %}F{% endif %}`, "F", "a sequence is not, even when empty"},
		// defined on a non-mapping
		{`{% if messages[0].role.nosuchfield is defined %}T{% else %}F{% endif %}`, "F",
			"a field read from a string is not defined"},
		{`{% if messages[0].role is defined %}T{% else %}F{% endif %}`, "T", "a real field is"},
	}
	for _, c := range cases {
		tpl, err := Parse(c.src)
		if err != nil {
			t.Fatalf("Parse(%q): %v", c.src, err)
		}
		got, err := tpl.Render(msgs, Inputs{})
		if err != nil {
			t.Fatalf("Render(%q): %v", c.src, err)
		}
		if got != c.want {
			t.Errorf("%q = %q, want %q (%s)", c.src, got, c.want, c.why)
		}
	}
}

// TestNotNestsAndReportsErrors covers the two properties of the `not` parser.
//
// `not` recurses into itself rather than into the next precedence level, which is what makes
// `not not x` parse. Parsing the next level instead would accept only one negation and then
// leave the second `not` to be read as a name, so `{% if not not true %}` would render the wrong
// branch rather than fail loudly.
//
// And an operand that cannot be parsed has to be reported. A parser that swallowed the error and
// built the node around a nil operand would accept `{% if not %}` and then either render
// something arbitrary or panic during evaluation.
//
// Neither had a test: every case in the corpus uses a single negation with a valid operand.
func TestNotNestsAndReportsErrors(t *testing.T) {
	// Odd numbers of negations flip the answer; even numbers restore it.
	for _, c := range []struct{ src, want string }{
		{`{% if not false %}T{% else %}F{% endif %}`, "T"},
		{`{% if not not true %}T{% else %}F{% endif %}`, "T"},
		{`{% if not not not false %}T{% else %}F{% endif %}`, "T"},
		{`{% if not true %}T{% else %}F{% endif %}`, "F"},
		{`{% if not not false %}T{% else %}F{% endif %}`, "F"},
	} {
		tpl, err := Parse(c.src)
		if err != nil {
			t.Fatalf("Parse(%q): %v", c.src, err)
		}
		got, err := tpl.Render(nil, Inputs{})
		if err != nil {
			t.Fatalf("Render(%q): %v", c.src, err)
		}
		if got != c.want {
			t.Errorf("%q = %q, want %q", c.src, got, c.want)
		}
	}

	// A missing operand is a parse error, not an accepted node.
	for _, src := range []string{
		`{% if not %}{% endif %}`,
		`{% if not ( %}{% endif %}`,
		`{% if not == %}{% endif %}`,
	} {
		if _, err := Parse(src); err == nil {
			t.Errorf("Parse(%q) accepted a negation with no operand", src)
		}
	}
}

// TestDanglingBooleanOperatorIsARefusal covers the right operand of and/or when it cannot be
// parsed at all.
//
// The operator loop collects the right side and propagates its error. Dropping that propagation
// makes `{% if true or %}` PARSE successfully -- measured -- with a nil right operand, so the
// template is accepted and the failure moves to render time, where a nil expr reaches the
// evaluator's type switch. A template error at parse time is what the package promises; the same
// defect one stage later is not.
//
// The cases cover a missing operand, an unclosed group and a stray keyword on either side of the
// operator, so an implementation that only checked for one shape cannot pass.
func TestDanglingBooleanOperatorIsARefusal(t *testing.T) {
	for _, src := range []string{
		`{% if true or %}{% endif %}`,
		`{% if true and %}{% endif %}`,
		`{% if true or ( %}{% endif %}`,
		`{% if true and ( %}{% endif %}`,
		`{{ true or }}`,
		`{{ true and }}`,
	} {
		if _, err := Parse(src); err == nil {
			t.Errorf("Parse(%q) accepted a dangling operator", src)
		}
	}

	// The chains themselves still build left to right, and short-circuit as before.
	for _, c := range []struct{ src, want string }{
		{`{% if true and true and false %}T{% else %}F{% endif %}`, "F"},
		{`{% if false or false or true %}T{% else %}F{% endif %}`, "T"},
		{`{% if false and true or true %}T{% else %}F{% endif %}`, "T"},
		{`{% if true or false and false %}T{% else %}F{% endif %}`, "T"},
	} {
		tpl, err := Parse(c.src)
		if err != nil {
			t.Fatalf("Parse(%q): %v", c.src, err)
		}
		got, err := tpl.Render(nil, Inputs{})
		if err != nil {
			t.Fatalf("Render(%q): %v", c.src, err)
		}
		if got != c.want {
			t.Errorf("%q = %q, want %q", c.src, got, c.want)
		}
	}
}

// TestDanglingBinaryOperatorIsARefusal covers the same defect one precedence level down.
//
// Every binary level collects its right operand and propagates the error: or/and (covered
// above), the comparisons, and concatenation. Dropping the propagation at any of them makes the
// template parse with a nil child, and the nil reaches the evaluator's type switch at render
// time instead of being refused here.
//
// The cases list one operator from each level and both sides, so a fix that special-cased one
// loop cannot pass. `in` is included because it is parsed by the comparison level but spelled
// like a filter would be, which is where a hand-written operand check tends to miss it.
func TestDanglingBinaryOperatorIsARefusal(t *testing.T) {
	operators := []string{"or", "and", "==", "!=", "in", "+"}
	for _, op := range operators {
		for _, src := range []string{
			`{% if a ` + op + ` %}{% endif %}`,
			`{{ a ` + op + ` }}`,
		} {
			if _, err := Parse(src); err == nil {
				t.Errorf("Parse(%q) accepted a dangling %q", src, op)
			}
		}
	}
}

// TestTojsonIndentReportsTheUndefinedName covers the diagnostic quality of a failing indent
// argument.
//
// Both the keyword and the positional form evaluate their argument before converting it. If the
// evaluation error is dropped, the empty value it produces fails the conversion instead, and the
// render reports `tojson indent "" is not a number` -- measured -- where the template's real
// problem is that `indent=nosuchvar` names a variable that does not exist. A template author
// reading the first message looks for a number where there is a typo.
//
// The test asserts the message names the undefined variable, which only holds if the evaluation
// error is propagated rather than replaced.
func TestTojsonIndentReportsTheUndefinedName(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}
	for _, c := range []struct{ name, src string }{
		{"keyword form", `{{ messages|tojson(indent=nosuchvar) }}`},
		{"positional form", `{{ messages|tojson(nosuchvar) }}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			_, err = tpl.Render(msgs, Inputs{})
			if err == nil {
				t.Fatalf("%q rendered without error", c.src)
			}
			if !strings.Contains(err.Error(), "undefined variable") {
				t.Errorf("error %q does not name the undefined variable", err)
			}
			if !strings.Contains(err.Error(), "nosuchvar") {
				t.Errorf("error %q does not quote the name the template used", err)
			}
		})
	}

	// A real indent still applies, so the error path is not standing in for a broken filter.
	tpl, err := Parse(`{{ messages|tojson(indent=2) }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	out, err := tpl.Render(msgs, Inputs{})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if !strings.Contains(out, "\n") {
		t.Errorf("tojson(indent=2) produced %q, which is not indented", out)
	}
}

// TestLexerErrorsReachTheTemplateAuthor covers the two ways an expression can fail to lex.
//
// An empty expression and an unexpected character both have to be reported with the text that
// caused them. The messages are the only thing a template author sees, and the alternative --
// an empty token list or a silently dropped character -- turns a typo into a template that
// renders something plausible and wrong.
//
// The unparseable characters here are the ones a chat format might use for something else: `$`
// and `~` appear in shell syntax and Jinja's own concatenation operator respectively. `~` is
// worth a case of its own because it is a real Jinja operator this subset does not implement, so
// the message has to name it rather than treat the expression as empty.
func TestLexerErrorsReachTheTemplateAuthor(t *testing.T) {
	cases := []struct {
		src  string
		want string
	}{
		{`{{ }}`, "empty expression"},
		{`{% if %}{% endif %}`, "empty expression"},
		{`{{ $ }}`, `unexpected "$"`},
		{`{{ a $ b }}`, `unexpected "$"`},
		{`{% if $ %}{% endif %}`, `unexpected "$"`},
		// Jinja's concatenation operator, which this subset does not implement.
		{`{{ 'a' ~ 'b' }}`, `unexpected "~"`},
	}
	for _, c := range cases {
		_, err := Parse(c.src)
		if err == nil {
			t.Errorf("Parse(%q) accepted it", c.src)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("Parse(%q) error %q does not say %q", c.src, err, c.want)
		}
		// The message quotes the expression itself, which is how a reader finds the position
		// in a template they did not write in this session.
		if c.want != "empty expression" && !strings.Contains(err.Error(), "in expression") {
			t.Errorf("Parse(%q) error %q does not name the expression", c.src, err)
		}
	}

	// A valid expression still lexes, so the refusal is about the input rather than about the
	// lexer being broken.
	tpl, err := Parse(`{{ 'a' + 'b' }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got, err := tpl.Render(nil, Inputs{}); err != nil || got != "ab" {
		t.Errorf("rendering a valid expression = %q, %v", got, err)
	}
}

// TestDanglingMinusIsARefusal covers the arithmetic level's right operand.
//
// Subtracting a missing operand parses successfully when the operand's error is dropped --
// measured -- and the template then fails at render time with "internal error: unknown
// expression <nil>". That message says the engine has a bug, which is the wrong thing to tell a
// template author whose expression is incomplete.
//
// The arithmetic operator is the one a chat template is least likely to use, which is why nothing
// reached it; the same loop shape is used by every other binary level, so a refusal here is
// consistent with the rest.
func TestDanglingMinusIsARefusal(t *testing.T) {
	for _, src := range []string{
		`{{ 1 - }}`,
		`{% if 1 - %}{% endif %}`,
		`{{ a - }}`,
		`{{ (1 - ) }}`,
	} {
		_, err := Parse(src)
		if err == nil {
			t.Errorf("Parse(%q) accepted a dangling minus", src)
			continue
		}
		// A parse-time refusal, not an internal error surfaced later.
		if strings.Contains(err.Error(), "internal error") {
			t.Errorf("Parse(%q) reported an internal error: %v", src, err)
		}
	}

	// Subtraction still works, and is left-associative: 10-3-2 is 5, not 9.
	tpl, err := Parse(`{{ 10 - 3 - 2 }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, err := tpl.Render(nil, Inputs{})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if got != "5" {
		t.Errorf("10 - 3 - 2 = %q, want 5 (left-associative)", got)
	}
}

// TestFilterNameMustBeAnIdentifier covers the check after '|'.
//
// A non-name after the pipe has to be refused at parse time. Without the check the parser reads
// the next token as the filter's name whatever it is, looks it up in supportedFilters, and reports
// "unsupported template filter \"1\"" -- which names a filter the author did not write and hides
// the malformed expression.
//
// The three shapes are the ones a typo produces: a number, a missing name, and a character the
// lexer already refuses with a different message (kept as a control, so the assertion is about
// this check rather than about the lexer).
func TestFilterNameMustBeAnIdentifier(t *testing.T) {
	msgs := []Message{{Role: "user", Content: " hi "}}

	// Only spellings that fail isIdentifier reach this check. `_` and a quoted 'x' are both
	// identifiers by their text -- the check looks at the token's content, not its kind -- so
	// they are reported as unknown filters instead, which is the case below.
	for _, src := range []string{`{{ x | 1 }}`, `{{ x | }}`, `{{ x | 1.5 }}`, `{{ x | 2a }}`} {
		_, err := Parse(src)
		if err == nil {
			t.Errorf("Parse(%q) accepted a non-identifier filter name", src)
			continue
		}
		if !strings.Contains(err.Error(), "expected a filter name") {
			t.Errorf("Parse(%q) error %q does not say a filter name was expected", src, err)
		}
	}

	// A real filter still resolves, and an unknown but well-formed name is reported by name.
	if got := render(t, `{{ messages[0].content | trim }}`, msgs, Inputs{}); got != "hi" {
		t.Errorf("trim gave %q, want \"hi\"", got)
	}
	_, err := Parse(`{{ messages | nosuchfilter }}`)
	if err == nil || !strings.Contains(err.Error(), `"nosuchfilter"`) {
		t.Errorf("an unknown filter name was not reported by name: %v", err)
	}
}

// TestKwargNameMustBeAnIdentifier covers the lookahead that decides between a keyword argument
// and a positional one.
//
// The lookahead has three conditions and the identifier check is the one that stops `1=2` from
// being read as a keyword argument named "1". Without it that call parses successfully --
// measured -- and the argument is stored under a key nothing reads, so the template renders as
// if the argument had not been written. A refusal that names the missing ')' is the honest
// answer to a call the subset cannot express.
//
// The other two conditions are already guarded: a bare name with no '=' is positional, and a
// keyword whose value fails to parse is reported.
func TestKwargNameMustBeAnIdentifier(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}

	// A well-formed keyword argument and a positional one both parse. `tojson(indent)` is the
	// case that separates the two lookahead conditions: `indent` is a legal name, so only the
	// '=' test stops it being read as a keyword whose value is whatever follows.
	for _, src := range []string{
		`{{ messages|tojson(indent=2) }}`,
		`{{ messages|tojson(2) }}`,
		`{{ messages|tojson(indent) }}`,
		`{{ messages|tojson }}`,
	} {
		if _, err := Parse(src); err != nil {
			t.Errorf("Parse(%q): %v", src, err)
		}
	}

	// A numeric name is not a keyword argument. The call is then unclosed, and the message that
	// reports it depends on where the operand search stopped: a plain number leaves the parser
	// looking for ')' or ',', while "1.5" is lexed as 1 . 5 and fails as a field access. Both
	// are refusals; what the test insists on is that neither parses.
	for _, src := range []string{`{{ messages|tojson(1=2) }}`, `{{ messages|tojson(1.5=2) }}`} {
		if _, err := Parse(src); err == nil {
			t.Errorf("Parse(%q) accepted a numeric keyword name", src)
		}
	}

	// A keyword with no value fails, and the message names the expression rather than the
	// argument list: `indent=` is a truncated expression, and saying so points at the token to
	// fix. Swallowing that error gives "missing ')' in filter arguments" instead, which sends
	// the reader looking for a bracket.
	if _, err := Parse(`{{ messages|tojson(indent=) }}`); err == nil {
		t.Error("Parse accepted a keyword with no value")
	} else if !strings.Contains(err.Error(), "expression") {
		t.Errorf("error %q does not point at the truncated expression", err)
	}

	// The rendered keyword still reaches the filter, so the positive case is not vacuous.
	tpl, err := Parse(`{{ messages|tojson(indent=2) }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	out, err := tpl.Render(msgs, Inputs{})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if !strings.Contains(out, "\n") {
		t.Errorf("indent=2 did not indent: %q", out)
	}
}

// TestEvaluationRefusesNonNumericOperands covers the two conversions that can fail at render
// time and neither had a test.
//
// A slice bound and an arithmetic operand both come from an expression, so either can be a
// string or a mapping the template supplied. Refusing names the operand, which is what a
// template author needs; converting with a zero instead would render a silently wrong number --
// `messages['a':]` becoming the whole list, or `'a' - 1` becoming -1.
//
// The mapping case is worth seeing: a container carries empty text, so the message quotes "" for
// it, and the test pins that rather than assuming a name is printed.
func TestEvaluationRefusesNonNumericOperands(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "x"}}

	for _, c := range []struct{ name, src, want string }{
		{"slice bound", `{{ messages['a':] }}`, `slice start "a" is not a number`},
		{"string operand", `{{ 'a' - 1 }}`, `arithmetic on non-numeric "a" and "1"`},
		{"mapping operand", `{{ messages - 1 }}`, `arithmetic on non-numeric`},
		{"number after mapping", `{{ 1 - messages }}`, `arithmetic on non-numeric`},
	} {
		t.Run(c.name, func(t *testing.T) {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			out, err := tpl.Render(msgs, Inputs{})
			if err == nil {
				t.Fatalf("%q rendered %q, want a refusal", c.src, out)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not say %q", err, c.want)
			}
			if out != "" {
				t.Errorf("%q produced %q alongside the error", c.src, out)
			}
		})
	}

	// Numeric operands still compute, so the refusal is about the operands rather than about
	// arithmetic.
	if got := render(t, `{{ 7 - 2 }}`, msgs, Inputs{}); got != "5" {
		t.Errorf("7 - 2 = %q, want 5", got)
	}
	// And a numeric slice bound still slices.
	tpl, err := Parse(`{% for m in messages[0:] %}{{ m.role }}{% endfor %}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got, err := tpl.Render(msgs, Inputs{}); err != nil || got != "user" {
		t.Errorf("messages[0:] = %q, %v", got, err)
	}
}

// TestNodeEvaluationErrorsPropagate covers the error returns of set and for.
//
// A statement that fails while computing its operand must stop the render and report the cause.
// The two cases use an expression that fails at render time -- a non-numeric slice bound -- rather
// than a parse error, because a parse error never reaches these nodes.
//
// The set case with a statement that assigns successfully is the other half: the failure happens
// before the assignment, so the name must not be bound. A template that continued with an empty
// binding would render a prompt with a hole in it, which is the failure this package's design
// notes single out.
func TestNodeEvaluationErrorsPropagate(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "x"}}

	for _, c := range []struct{ name, src string }{
		{"set", `{% set x = messages['a':] %}{{ x }}`},
		{"for", `{% for m in messages['a':] %}{{ m.role }}{% endfor %}`},
		{"if body", `{% if messages %}x{% endif %}{% set y = messages['a':] %}{{ y }}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			out, err := tpl.Render(msgs, Inputs{})
			if err == nil {
				t.Fatalf("%q rendered %q, want the operand's error", c.src, out)
			}
			if !strings.Contains(err.Error(), "slice start") {
				t.Errorf("error %q does not carry the operand's failure", err)
			}
		})
	}

	// A set whose operand succeeds still binds, so the refusal is about the operand.
	tpl, err := Parse(`{% set x = messages[0].content %}{{ x }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got, err := tpl.Render(msgs, Inputs{}); err != nil || got != "x" {
		t.Errorf("set with a good operand = %q, %v", got, err)
	}
}

// TestTojsonMatchesPythonsEscaping pins the escaping to what Jinja2's filter produces.
//
// A chat template interpolates a tool definition or a message straight into the prompt, so the
// spelling of every non-ASCII character is part of the prompt the model sees. Python's
// json.dumps defaults to ensure_ascii=True and Jinja2 inherits that, so a Chinese description
// arrives as \uXXXX and an emoji as a surrogate pair -- where Go's json.Marshal would leave the
// UTF-8 in place and spell the same text differently.
//
// The expected strings below were taken from jinja2 3.1.6, one rendering per case, not derived
// from this package's own helper:
//
//   - "中文"    -> "\u4e2d\u6587"
//   - "🎉"     -> "\ud83c\udf89"   (a surrogate pair, not \U0001f389)
//   - "a\tb"   -> "a\tb"          (short escapes stay short)
//
// Both arms of the switch fail when broken: removing the BMP escape leaves the UTF-8 in place,
// and off-by-one in the surrogate pair changes the emoji.
func TestTojsonMatchesPythonsEscaping(t *testing.T) {
	tpl, err := Parse(`{{ messages[0].content|tojson }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	cases := []struct{ in, want string }{
		{"中文", `"\u4e2d\u6587"`},
		{"🎉", `"\ud83c\udf89"`},
		{"a\tb", `"a\tb"`},
		{`quote"x`, `"quote\"x"`},
		{`back\slash`, `"back\\slash"`},
		{"nl\nx", `"nl\nx"`},
		// ASCII stays literal: escaping it would be valid JSON and wrong for this filter.
		{"plain", `"plain"`},
	}
	for _, c := range cases {
		got, err := tpl.Render([]Message{{Role: "user", Content: c.in}}, Inputs{})
		if err != nil {
			t.Fatalf("Render(%q): %v", c.in, err)
		}
		if got != c.want {
			t.Errorf("tojson(%q) = %s, want %s", c.in, got, c.want)
		}
	}
}

// TestParseErrorsNameTheOffendingTag covers the three parse refusals whose message carries the tag or
// header it rejected.
//
// "unexpected closing statement" tells a template author that something is unbalanced and not which
// line to look at, and a file with ten blocks leaves them to find it. The messages already name the
// statement; nothing asserted the name, so dropping it from the format string passed.
func TestParseErrorsNameTheOffendingTag(t *testing.T) {
	cases := []struct{ src, want string }{
		// A closing tag with nothing open.
		{`{% endfor %}`, "unexpected {% endfor %}"},
		{`{% endif %}`, "unexpected {% endif %}"},
		// A statement this parser does not implement, quoted so a typo is visible.
		{`{% macro x %}`, `unsupported template statement "macro"`},
		// A for tag with no in clause: the header is what has to be corrected.
		{`{% for %}`, "for  %} has no 'in' clause"},
		{`{% for x %}{% endfor %}`, "for x %} has no 'in' clause"},
	}
	for _, c := range cases {
		_, err := Parse(c.src)
		if err == nil {
			t.Errorf("Parse(%q) accepted a malformed statement", c.src)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("Parse(%q) error = %q, want it to contain %q", c.src, err, c.want)
		}
	}
}

// TestLexerErrorsLocateTheProblem covers the message for a tag that is opened and never closed.
//
// It carries two things a reader needs: which of the three delimiters was opened, and where. A
// template with several tags makes the offset the difference between finding the line and scanning
// the file, and the delimiter tells them whether to look for a brace or a percent sign.
//
// Nothing asserted either: replacing the format with "template has an unclosed tag" passed.
//
// The neighbouring branch, "template ends inside %q", is unreachable and deliberately left untested:
// the loop that finds a delimiter stops at j+1 < len(src), so by the time control reaches that check
// the byte after the brace is guaranteed to exist.
func TestLexerErrorsLocateTheProblem(t *testing.T) {
	cases := []struct{ src, want string }{
		{`a{% set x`, `unclosed "{%" at offset 1`},
		{`{{ x`, `unclosed "{{" at offset 0`},
		{`x{# note`, `unclosed "{#" at offset 1`},
		// The offset is not always zero or one: it is wherever the tag starts.
		{`abcdef{{ x`, `unclosed "{{" at offset 6`},
	}
	for _, c := range cases {
		_, err := Parse(c.src)
		if err == nil {
			t.Errorf("Parse(%q) accepted an unclosed tag", c.src)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("Parse(%q) error = %q, want it to contain %q", c.src, err, c.want)
		}
	}
}

// TestUnclosedCallNamesTheFunction covers the message for a call whose parenthesis never closes.
//
// The name is what a reader needs: `range(` and `tojson(` fail the same way and are fixed in
// different places, and a template with both would otherwise send them to the wrong one. Removing
// the name from the format string passed every test.
func TestUnclosedCallNamesTheFunction(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ range(1, 2 }}`, `call to "range" is not closed`},
		{`{{ dict(1, 2 }}`, `call to "dict" is not closed`},
	}
	for _, c := range cases {
		_, err := Parse(c.src)
		if err == nil {
			t.Errorf("Parse(%q) accepted an unclosed call", c.src)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("Parse(%q) error = %q, want it to contain %q", c.src, err, c.want)
		}
	}
}

// TestTrailingTokensShowWhatWasParsed covers the message for an expression followed by more tokens.
//
// It prints the token it stopped at and the whole expression, which is what makes a typo visible:
// "unexpected token" alone leaves the author comparing their text against the grammar. The two
// together read as "this is where I stopped, and this is the span I had consumed".
func TestTrailingTokensShowWhatWasParsed(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ 1 2 }}`, `unexpected "2" after expression "1 2"`},
		{`{{ x y }}`, `unexpected "y" after expression "x y"`},
	}
	for _, c := range cases {
		_, err := Parse(c.src)
		if err == nil {
			t.Errorf("Parse(%q) accepted trailing tokens", c.src)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("Parse(%q) error = %q, want it to contain %q", c.src, err, c.want)
		}
	}
}

// TestEvalErrorsNameTheVariableAndField covers two render-time messages whose data went untested.
//
// A for loop over a non-sequence names the loop variable: a template with two loops needs to know
// which one, and the message is the only thing that says. The attribute form on a non-mapping names
// the field being read, for the same reason.
//
// Both messages carried the name and nothing asserted it; replacing either format with a sentence
// passed every test.
func TestEvalErrorsNameTheVariableAndField(t *testing.T) {
	t.Run("for over a non-sequence", func(t *testing.T) {
		tpl, err := Parse(`{% for item in 1 %}{{ item }}{% endfor %}`)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		_, err = tpl.Render(nil, Inputs{})
		if err == nil {
			t.Fatal("rendering a for over a number succeeded")
		}
		if !strings.Contains(err.Error(), "iterates a non-sequence") {
			t.Errorf("error %q does not say what went wrong", err)
		}
		if !strings.Contains(err.Error(), "item") {
			t.Errorf("error %q does not name the loop variable", err)
		}
	})

	t.Run("attribute of a non-mapping", func(t *testing.T) {
		// A message's content is a string, so reading .something from it is the case.
		tpl, err := Parse(`{{ messages[0].content.nosuchfield }}`)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		_, err = tpl.Render([]Message{{Role: "user", Content: "hi"}}, Inputs{})
		if err == nil {
			t.Fatal("reading a field from a string succeeded")
		}
		if !strings.Contains(err.Error(), "non-mapping") {
			t.Errorf("error %q does not say what went wrong", err)
		}
		if !strings.Contains(err.Error(), "nosuchfield") {
			t.Errorf("error %q does not name the field", err)
		}
	})
}

// TestIndexErrorsShowTheIndex covers the two messages a bad sequence index produces.
//
// A non-numeric index prints the text it got, and an out-of-range one prints the index and the length
// it exceeded. Both are what a template author compares against, and the non-numeric case had no test
// at all -- removing the quoted text from the format passed everything.
func TestIndexErrorsShowTheIndex(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}

	t.Run("a non-numeric index", func(t *testing.T) {
		tpl, err := Parse(`{{ messages['x'] }}`)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		_, err = tpl.Render(msgs, Inputs{})
		if err == nil {
			t.Fatal("indexing a sequence with a string succeeded")
		}
		// The quoted text, so a template with several indexes says which one is wrong.
		if !strings.Contains(err.Error(), `"x"`) {
			t.Errorf("error %q does not show the index it got", err)
		}
		if !strings.Contains(err.Error(), "is not a number") {
			t.Errorf("error %q does not say what the index should be", err)
		}
	})
}

// TestRenderIsSafeForConcurrentUse pins the concurrency contract the type documents.
//
// The callers share one Template across goroutines: engine.ChatRenderer is built once per
// model and the HTTP server renders from every request handler that arrives. Nothing in the
// type changes after Parse today -- the node tree is read-only and Render builds a fresh
// scope and buffer -- but that is a property of the code rather than of an assertion, and a
// field added later that a render writes to would be a data race in production with no test
// to notice.
//
// Run with -race, which is what makes this a check rather than a smoke test: a shared buffer
// or a cached scope fails the detector here. The output is compared as well, so a render that
// were merely race-free but produced another goroutine's text would still fail.
func TestRenderIsSafeForConcurrentUse(t *testing.T) {
	// A template that exercises a loop, a conditional, a set and a filter, so a shared scope
	// would have something to corrupt rather than only a shared buffer.
	src := `{% set greet = "hi" %}` +
		`{% for m in messages %}{{ greet }} {{ m.role }}: {{ m.content | trim }}` + "\n" +
		`{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}`
	tmpl, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	const goroutines = 32
	var wg sync.WaitGroup
	failures := make(chan error, goroutines)
	for i := range goroutines {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// Each goroutine renders its own content, so a value crossing between them is
			// visible as a mismatch rather than as a lucky match.
			content := fmt.Sprintf("  message %d  ", i)
			got, err := tmpl.Render(
				[]Message{{Role: "user", Content: content}},
				Inputs{AddGenerationPrompt: true})
			if err != nil {
				failures <- fmt.Errorf("goroutine %d: %w", i, err)
				return
			}
			want := fmt.Sprintf("hi user: message %d\nassistant:", i)
			if got != want {
				failures <- fmt.Errorf("goroutine %d rendered %q, want %q", i, got, want)
			}
		}(i)
	}
	wg.Wait()
	close(failures)
	for err := range failures {
		t.Error(err)
	}
}

// TestPlusConcatenatesRatherThanAdds pins what the + operator means here.
//
// It concatenates, so "1 + 2" renders "12". That is the documented behaviour -- doc.go lists
// "+ concatenation" among the supported constructs -- and it differs from Jinja2, where the
// same expression renders "3". Nothing tested either reading: the only arithmetic tests
// covered subtraction, so an evaluator that started adding numbers would have changed what
// every prompt says without failing anything.
//
// The mixed-operand cases are the ones real templates write: a literal marker plus a message
// field, which is why no type check happens before joining.
func TestPlusConcatenatesRatherThanAdds(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}
	cases := []struct{ name, src, want string }{
		{"two numbers", `{{ 1 + 2 }}`, "12"},
		{"three numbers", `{{ 1 + 2 + 3 }}`, "123"},
		{"two strings", `{{ "a" + "b" }}`, "ab"},
		{"literal and field", `{% for m in messages %}{{ "role: " + m.role }}{% endfor %}`, "role: user"},
		{"number and field", `{% for m in messages %}{{ loop.index + m.content }}{% endfor %}`, "1hi"},
		{"a set variable", `{% set x = 1 %}{{ x + 2 }}`, "12"},
		{"subtraction still subtracts", `{{ 1 - 2 }}`, "-1"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := render(t, c.src, msgs, Inputs{}); got != c.want {
				t.Errorf("%s rendered %q, want %q", c.src, got, c.want)
			}
		})
	}
}

// TestArithmeticDetectsOverflow covers the one arithmetic operator this package implements.
//
// Subtraction is checked rather than wrapped. The operands are arbitrary text: a value from
// the message list or a set literal reaches Atoi, so a number near the int64 limits can arrive.
// Measured before the check, "-9223372036854775808" minus 1 rendered "9223372036854775807" and
// MaxInt64 minus -1 wrapped the other way -- a template that subtracted to index a list would
// then read whatever element happened to sit at the wrapped position, with nothing to notice.
//
// The boundaries on both sides are asserted, including the two values just inside them, so a
// check that rejected everything near a limit would fail here rather than pass.
func TestArithmeticDetectsOverflow(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "a"}}

	render := func(t *testing.T, a, b string) (string, error) {
		t.Helper()
		tpl, err := Parse(fmt.Sprintf(`{%% set a = "%s" %%}{{ a - (%s) }}`, a, b))
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		return tpl.Render(msgs, Inputs{})
	}

	for _, c := range []struct{ name, a, b string }{
		{"max minus a negative", "9223372036854775807", "-1"},
		{"min minus one", "-9223372036854775808", "1"},
		{"min minus a larger positive", "-9223372036854775808", "9223372036854775807"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := render(t, c.a, c.b)
			if err == nil {
				t.Fatalf("%s - %s did not report an overflow", c.a, c.b)
			}
			if !strings.Contains(err.Error(), "overflow") {
				t.Errorf("error %q does not say what happened", err)
			}
			// Both operands, so a template author can find the expression.
			for _, want := range []string{c.a, c.b} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not state the operand %q", err, want)
				}
			}
		})
	}

	// The values that are representable still work, on both sides of zero and at the limits
	// themselves. A guard that wrapped or rejected either boundary would fail here.
	for _, c := range []struct{ name, a, b, want string }{
		{"ordinary", "10", "3", "7"},
		{"both negative", "-10", "-3", "-7"},
		{"zero", "0", "0", "0"},
		{"max minus one", "9223372036854775807", "1", "9223372036854775806"},
		{"min plus one minus one", "-9223372036854775807", "1", "-9223372036854775808"},
		{"max minus a large negative", "9223372036854775806", "-1", "9223372036854775807"},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := render(t, c.a, c.b)
			if err != nil {
				t.Fatalf("%s - %s was rejected: %v", c.a, c.b, err)
			}
			if got != c.want {
				t.Errorf("%s - %s = %q, want %q", c.a, c.b, got, c.want)
			}
		})
	}
}

// TestParseErrorsNameTheOffset covers where each parse error happened.
//
// A message that says only what was wrong leaves the reader counting tags: a chat template is
// tens of lines, and "unsupported template statement "bogus"" does not say which one. The
// lexer now records each tag's byte position, and the two block errors carry both ends -- the
// opener and the closer -- because "closed by the wrong tag" is a relationship between two
// places and naming one of them is half an answer.
//
// The expected offsets are computed from the source rather than written as numbers, so a
// change to how offsets are counted shows up as a mismatch here rather than as a set of
// assertions that were updated to match.
func TestParseErrorsNameTheOffset(t *testing.T) {
	for _, c := range []struct {
		name     string
		src      string
		wantIdx  []string // substrings whose index in src the message must report
		wantWord string
	}{
		{
			name:     "an unsupported statement",
			src:      "line1\nline2\n{% bogus %}",
			wantIdx:  []string{"{% bogus %}"},
			wantWord: "unsupported template statement",
		},
		{
			name:     "a closer with no block",
			src:      "x\n{% endif %}",
			wantIdx:  []string{"{% endif %}"},
			wantWord: "with no open block",
		},
		{
			name:     "an if that never closes",
			src:      "aa\n{% if true %}",
			wantIdx:  []string{"{% if true %}"},
			wantWord: "is never closed",
		},
		{
			name:     "a for that never closes",
			src:      "\n{% for m in messages %}",
			wantIdx:  []string{"{% for m in messages %}"},
			wantWord: "is never closed",
		},
		{
			name:     "an if closed by a for",
			src:      "{% if x %}{% endfor %}",
			wantIdx:  []string{"{% if x %}", "{% endfor %}"},
			wantWord: "closed by",
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := Parse(c.src)
			if err == nil {
				t.Fatalf("Parse accepted %q", c.src)
			}
			msg := err.Error()
			if !strings.Contains(msg, c.wantWord) {
				t.Errorf("error %q does not contain %q", msg, c.wantWord)
			}
			for _, tag := range c.wantIdx {
				want := strconv.Itoa(strings.Index(c.src, tag))
				if !strings.Contains(msg, "offset "+want) {
					t.Errorf("error %q does not report offset %s (where %q begins in %q)",
						msg, want, tag, c.src)
				}
			}
		})
	}

	// A block closed by the wrong tag names both ends: the opener's offset and the closer's.
	t.Run("both ends of a mismatched pair", func(t *testing.T) {
		const src = "xx{% if a %}yy{% endfor %}"
		_, err := Parse(src)
		if err == nil {
			t.Fatalf("Parse accepted %q", src)
		}
		msg := err.Error()
		for _, tag := range []string{"{% if a %}", "{% endfor %}"} {
			want := strconv.Itoa(strings.Index(src, tag))
			if !strings.Contains(msg, want) {
				t.Errorf("error %q omits offset %s (where %q begins)", msg, want, tag)
			}
		}
	})
}

// TestToJSONIndentIsBounded covers the two failures a huge indent caused.
//
// The indent is multiplied by the nesting depth and the product becomes a strings.Repeat length,
// so it is not a formatting knob with a linear cost. Measured before the bound:
// tojson(9223372036854775807) panicked with "makeslice: len out of range", and tojson(100000)
// on four short messages produced 2.4 MB of output. A template is model-file content, so a
// crash and a memory amplification are both reachable by opening a file.
//
// Python's json.dumps has no bound and a template in the wild writes one digit, so the limit
// sits well above anything a formatting choice needs; the cases below assert both sides of it.
func TestToJSONIndentIsBounded(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "a"}}

	render := func(t *testing.T, arg string) (string, error) {
		t.Helper()
		tpl, err := Parse(`{{ messages | tojson` + arg + ` }}`)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		return tpl.Render(msgs, Inputs{})
	}

	// The values that must keep working, including the largest one allowed.
	for _, c := range []struct{ name, arg string }{
		{"no argument", ""},
		{"zero", "(0)"},
		{"a formatting choice", "(2)"},
		{"negative", "(-1)"},
		{"exactly the limit", "(1024)"},
		{"keyword form", "(indent=2)"},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := render(t, c.arg); err != nil {
				t.Errorf("tojson%s was refused: %v", c.arg, err)
			}
		})
	}

	// The values that must be refused rather than attempted.
	for _, c := range []struct{ name, arg string }{
		{"just past the limit", "(1025)"},
		{"large", "(100000)"},
		{"the value that used to panic", "(9223372036854775807)"},
		{"keyword form past the limit", "(indent=1025)"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := render(t, c.arg)
			if err == nil {
				t.Fatalf("tojson%s was accepted", c.arg)
			}
			// The message has to name the value and the limit, or the template author cannot
			// tell which of their numbers was wrong.
			if !strings.Contains(err.Error(), "1024") {
				t.Errorf("error %q does not state the limit", err)
			}
		})
	}
}

// TestSetEscapesALoopBody pins where an assignment made inside a for body is visible after the
// loop, which is the one place this package deliberately differs from Jinja.
//
// Jinja scopes a for body: an assignment there does not escape it, and a name first assigned
// inside a loop renders empty afterwards. This package writes to the outermost frame, so the
// value survives. The comment on scope records the measurement against jinja2 3.1.6 and says
// the difference is kept because no template in the corpus the package was written for depends
// on either reading -- Llama 3's assigns and reads are both inside the body, which the two
// agree on.
//
// Nothing tested that, so a change towards Jinja's rule would have been silent. This is the
// behaviour as it stands, not a claim that it is right: if the decision is ever revisited, the
// test should be changed in the same commit as the comment that explains it.
func TestSetEscapesALoopBody(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "a"}, {Role: "user", Content: "b"}}

	for _, c := range []struct {
		name, src, want string
	}{
		{
			name: "a value set before the loop is unchanged by it",
			src:  `{% set x = "before" %}{% for m in messages %}{% endfor %}{{ x }}`,
			want: "before",
		},
		{
			name: "a new name assigned in the body survives the loop",
			src:  `{% for m in messages %}{% set x = "inside" %}{% endfor %}{{ x }}`,
			want: "inside",
		},
		{
			name: "the body sees its own assignment",
			src:  `{% for m in messages %}{% set x = "in" %}{{ x }}{% endfor %}`,
			want: "inin",
		},
		{
			name: "the last iteration wins after the loop",
			src:  `{% set z = "a" %}{% for m in messages %}{% set z = m.content %}{{ z }}{% endfor %}-{{ z }}`,
			want: "ab-b",
		},
		{
			name: "an if body is not a scope either",
			src:  `{% if true %}{% set y = "iffed" %}{% endif %}{{ y }}`,
			want: "iffed",
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := render(t, c.src, msgs, Inputs{})
			if got != c.want {
				t.Errorf("%q rendered %q, want %q", c.src, got, c.want)
			}
		})
	}
}

// TestNestedLoopsAreBounded covers the work budget.
//
// Nesting multiplies the output by the message count per level, so a template of a few
// hundred bytes renders to an unbounded amount of work: measured over 8 messages, six
// nested for-loops took 420ms for 256 KiB and eight took 17.4s for 16 MiB. The template is
// model-file content, but the message list is request-controlled, so a caller could spend
// the server's time and memory on a prompt that could never fit the context.
//
// The output bound alone would not have caught it: reaching 16 MiB is what costs 17
// seconds, so a render that fails *at* the byte bound has already paid for it. The node
// budget stops the same shape in milliseconds, which is why the depth used here is one that
// used to run for tens of seconds.
func TestNestedLoopsAreBounded(t *testing.T) {
	msgs := make([]Message, 8)
	for i := range msgs {
		msgs[i] = Message{Role: "user", Content: "x"}
	}

	// A one-level template is the shape every published checkpoint uses; it must be
	// untouched by the budget, at a message count larger than a real conversation.
	t.Run("a real shape is unaffected", func(t *testing.T) {
		one, err := Parse(`{% for m in messages %}{{ m.content }}{% endfor %}`)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		out, err := one.Render(msgs, Inputs{})
		if err != nil {
			t.Fatalf("a single-level loop was refused: %v", err)
		}
		if want := strings.Repeat("x", len(msgs)); out != want {
			t.Errorf("rendered %q, want %q", out, want)
		}
	})

	for _, depth := range []int{8, 10} {
		t.Run(strconv.Itoa(depth)+" nested loops", func(t *testing.T) {
			var b strings.Builder
			b.WriteString("{% for m in messages %}")
			for range depth - 1 {
				b.WriteString("{% for m in messages %}")
			}
			b.WriteString("{{ m.content }}")
			for range depth - 1 {
				b.WriteString("{% endfor %}")
			}
			b.WriteString("{% endfor %}")

			tpl, err := Parse(b.String())
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			start := time.Now()
			out, err := tpl.Render(msgs, Inputs{})
			elapsed := time.Since(start)
			if err == nil {
				t.Fatalf("a %d-level nest rendered %d bytes without an error", depth, len(out))
			}
			if !errors.Is(err, errRenderTooLarge) {
				t.Errorf("error %v does not carry errRenderTooLarge", err)
			}
			// The bound exists to keep this cheap, so the test asserts it: 17 seconds was
			// the measured cost of the shape this replaces.
			if elapsed > 2*time.Second {
				t.Errorf("the render took %v to refuse, which is the cost the budget exists "+
					"to avoid", elapsed.Round(time.Millisecond))
			}
		})
	}
}

// TestRenderByteBoundIsReachableAndEnforced covers the second of the two bounds.
//
// maxRenderNodes catches a template whose nesting multiplies its work, and
// TestNestedLoopsAreBounded pins that. maxRenderBytes exists for the other shape: a
// template that stays well within the node budget and still emits something enormous. The
// comment on the constant names `tojson` over a large value as the example, and this is
// that case -- a single `{{ messages | tojson }}` node against a long message list produced
// more than 16 MiB, so the node count never came into it.
//
// Deleting the byte check left every package green, which is why it is worth a test: the
// bound is reachable, it is the only thing that stops this shape, and nothing proved it ran.
func TestRenderByteBoundIsReachableAndEnforced(t *testing.T) {
	// 400 messages of 64 KiB each: tojson over the list is roughly 26 MiB, and the whole
	// render is one node.
	const msgSize = 64 << 10
	content := strings.Repeat("x", msgSize)
	msgs := make([]Message, 400)
	for i := range msgs {
		msgs[i] = Message{Role: "user", Content: content}
	}
	if want := len(msgs) * msgSize; want <= maxRenderBytes {
		t.Fatalf("the fixture produces about %d bytes, not more than the %d bound, so it "+
			"could not exercise it", want, maxRenderBytes)
	}

	tpl, err := Parse(`{{ messages | tojson }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	out, err := tpl.Render(msgs, Inputs{})
	if err == nil {
		t.Fatalf("a render of %d bytes returned no error (bound is %d)", len(out), maxRenderBytes)
	}
	if !errors.Is(err, errRenderTooLarge) {
		t.Errorf("error %v does not carry errRenderTooLarge", err)
	}
	// The message has to name the bound, or a template author cannot tell which limit they
	// hit -- the node budget reports a node count, this one a byte count.
	if !strings.Contains(err.Error(), "16777216") {
		t.Errorf("error %q does not state the byte bound", err)
	}
}

// TestRenderModulo covers '%', which exists for Gemma's alternation guard.
//
// The parity test is the shape the template uses, so it is the one asserted first. The
// precedence case matters separately: '%' has to bind tighter than '-' or binary search from the
// left would turn "4 - 1 % 2" into "(4 - 1) % 2" -- 1 rather than 3.
func TestRenderModulo(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ 4 % 2 }}`, "0"},
		{`{{ 5 % 2 }}`, "1"},
		{`{{ 0 % 3 }}`, "0"},
		{`{{ 7 % 7 }}`, "0"},
		// Tighter than '-': 1 % 2 is 1, so 4 - 1 is 3. The other grouping gives 1.
		{`{{ 4 - 1 % 2 }}`, "3"},
		// Looser than a function call and an index: both bind first.
		{`{{ (3 + 1) % 3 }}`, "1"},
		// The form the Gemma template uses, over the loop counter.
		{`{% for m in messages %}{{ loop.index0 % 2 }}{% endfor %}`, "010"},
	}
	for _, tc := range cases {
		msgs := []Message{{Role: "user", Content: "a"}, {Role: "assistant", Content: "b"}, {Role: "user", Content: "c"}}
		if got := render(t, tc.src, msgs, Inputs{}); got != tc.want {
			t.Errorf("%q = %q, want %q", tc.src, got, tc.want)
		}
	}
}

// TestRenderModuloRejects covers the operands '%' will not take.
//
// Both refusals are deliberate rather than incidental. Dividing by zero has no answer, and a
// negative operand has two: Go's truncating division gives -1 for "-1 % 2", while Jinja gives 1
// as Python does. Answering with either would be a silent difference from the template's meaning,
// so the case is named as unsupported instead.
func TestRenderModuloRejects(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ 1 % 0 }}`, "modulo by zero"},
		{`{{ -1 % 2 }}`, "negative operand"},
		{`{{ 1 % 'a' }}`, "non-numeric"},
	}
	for _, tc := range cases {
		parsed, err := Parse(tc.src)
		if err != nil {
			t.Errorf("%q: Parse: %v", tc.src, err)
			continue
		}
		_, err = parsed.Render(nil, Inputs{})
		if err == nil {
			t.Errorf("%q rendered, want an error mentioning %q", tc.src, tc.want)
			continue
		}
		if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%q: error = %v, want it to mention %q", tc.src, err, tc.want)
		}
	}
}

// TestRaiseExceptionCarriesTheTemplate'sMessage covers the one global this package implements.
//
// raise_exception is not a filter or a value-producing function: its semantics are to abort with
// the text the template wrote, and templates use it as an assertion about the conversation --
// Gemma rejects a system role and a non-alternating history that way. Reporting "no function is
// implemented" instead left the operator with a fallback prompt and no idea why.
//
// The argument is a general expression rather than a literal, so it is evaluated: a template that
// builds the message from loop state has to produce the message it would have raised.
func TestRaiseExceptionCarriesTheTemplatesMessage(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ raise_exception('stop here') }}`, "template raised: stop here"},
		// A guarded call never runs, which is how the real templates use it.
		{`{% if false %}{{ raise_exception('no') }}{% endif %}ok`, "ok"},
		// The argument is evaluated like any other expression.
		{`{% for m in messages %}{{ raise_exception('bad role ' + m.role) }}{% endfor %}`,
			"template raised: bad role user"},
	}
	for _, tc := range cases {
		parsed, err := Parse(tc.src)
		if err != nil {
			t.Errorf("%q: Parse: %v", tc.src, err)
			continue
		}
		got, err := parsed.Render([]Message{{Role: "user", Content: "hi"}}, Inputs{})
		if strings.Contains(tc.want, "template raised") {
			if err == nil {
				t.Errorf("%q rendered %q, want an error", tc.src, got)
				continue
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("%q: error = %v, want it to contain %q", tc.src, err, tc.want)
			}
			continue
		}
		if err != nil {
			t.Errorf("%q: Render: %v", tc.src, err)
			continue
		}
		if got != tc.want {
			t.Errorf("%q = %q, want %q", tc.src, got, tc.want)
		}
	}

	// A different unimplemented call keeps the old behaviour: it is reported, not silently
	// treated as empty, because strftime_now would otherwise put a wrong date in a prompt.
	parsed, err := Parse(`{{ strftime_now('%Y') }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if _, err := parsed.Render(nil, Inputs{}); err == nil {
		t.Error("an unimplemented call other than raise_exception rendered instead of failing")
	} else if !strings.Contains(err.Error(), "no function is implemented") {
		t.Errorf("error = %v, want it to name the missing function", err)
	} else if !strings.Contains(err.Error(), "strftime_now") {
		// The message reached here says "want it to name the missing function" and checked only
		// that a function was missing. A template calling several of them gets one message, and
		// the name is the part that says which.
		t.Errorf("error = %v, want it to name strftime_now", err)
	}
}

// TestInlineConditionalAndStringTest covers the three constructs Gemma 3's template needs.
//
// Each was refused before this round, so the whole template fell back to the neutral transcript and
// every Gemma 3 conversation lost its <start_of_turn> markers. They are tested together because
// that is how they appeared: the template needs "is string" to tell a plain content string from a
// list of multimodal parts, "x if c else y" to emit the system prefix only on the first turn, and a
// parenthesised conditional inside a concatenation to combine the two.
func TestInlineConditionalAndStringTest(t *testing.T) {
	cases := []struct {
		src, want string
		msgs      []Message
		vars      map[string]value
		why       string
	}{
		{
			src:  `{{ 'yes' if flag else 'no' }}`,
			want: "yes",
			vars: map[string]value{"flag": boolValue(true)},
			why:  "the then arm is chosen when the condition holds",
		},
		{
			src:  `{{ 'yes' if flag else 'no' }}`,
			want: "no",
			vars: map[string]value{"flag": boolValue(false)},
			why:  "and the else arm when it does not",
		},
		{
			// The parenthesised form is what Gemma 3 writes, and it failed separately: the
			// grouping parser stopped at "if" and reported a missing parenthesis.
			src:  `{{ 'a' + (p if flag else '') }}`,
			want: "aZ",
			vars: map[string]value{"flag": boolValue(true), "p": strValue("Z")},
			why:  "a conditional inside parentheses concatenates like any other value",
		},
		{
			src:  `{{ (p if flag else '') + 'a' }}`,
			want: "a",
			vars: map[string]value{"flag": boolValue(false), "p": strValue("Z")},
			why:  "and the empty arm contributes nothing",
		},
		{
			src:  `{%- if messages[0]['content'] is string -%}plain{%- else -%}parts{%- endif -%}`,
			want: "plain",
			msgs: []Message{{Role: "user", Content: "hello"}},
			why:  "a string content takes the string branch",
		},
		{
			src:  `{%- if 'x' is string -%}y{%- endif -%}`,
			want: "y",
			why:  "a literal is a string too",
		},
		{
			// The branch Gemma 3 takes for multimodal content: a list is not a string, so the
			// template enters its per-part loop instead of emitting the value directly.
			src:  `{%- if v is string -%}str{%- else -%}not{%- endif -%}`,
			want: "not",
			vars: map[string]value{"v": listValue([]value{strValue("a")})},
			why:  "a sequence is not a string",
		},
	}
	for _, tc := range cases {
		// The whole template is parsed, so the src is exercised through the same path a checkpoint's
		// template takes, but it is evaluated against a hand-built scope: Inputs exposes only the
		// four names a checkpoint's template is given, and the cases need their own bindings.
		if _, err := Parse(tc.src); err != nil {
			t.Errorf("%q: Parse: %v (%s)", tc.src, err, tc.why)
			continue
		}
		vars := make(map[string]value, len(tc.vars)+1)
		for k, v := range tc.vars {
			vars[k] = v
		}
		if len(tc.msgs) > 0 {
			vars["messages"] = messagesValue(tc.msgs)
		}
		sc := newScope(vars)
		var r renderer
		parsed, _ := Parse(tc.src)
		if err := r.renderNodes(parsed.nodes, sc); err != nil {
			t.Errorf("%q: render: %v (%s)", tc.src, err, tc.why)
			continue
		}
		got := r.buf.String()
		if got != tc.want {
			t.Errorf("%q = %q, want %q (%s)", tc.src, got, tc.want, tc.why)
		}
	}

	// The shorter Jinja form has no else arm and yields undefined; this package cannot express
	// that, so it is refused rather than guessed at.
	if _, err := Parse(`{{ 'a' if flag }}`); err == nil {
		t.Error("a conditional without else parsed; Jinja's shorter form yields an undefined " +
			"value and no template here relies on one")
	}
}
