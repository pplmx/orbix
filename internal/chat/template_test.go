package chat

import (
	"encoding/json"
	"strings"
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
		`{{ 'a' if x else 'b' }}`,                       // inline if-expression
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
	if !strings.Contains(err.Error(), "offset") && !strings.Contains(err.Error(), "unclosed") {
		t.Errorf("error %q does not name a position", err)
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
// with ~, and the is-tests other than "is defined" -- because "we report what we do not
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
		`{% set x = y if z else w %}`,
		`{{ x if y else z }}`,
		`{% for k, v in d.items() %}{{ k }}{% endfor %}`,
		`{% for x in y %}{% else %}none{% endfor %}`,
		`{{ loop.cycle('a', 'b') }}`,
		`{{ 'a' ~ 'b' }}`,
		`{{ x is string }}`,
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
	for _, want := range []string{"Paris.", "And of Italy?", "<|im_start|>assistant"} {
		if !strings.Contains(got, want) {
			t.Errorf("prompt is missing %q:\n%s", want, got)
		}
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
		// An if closed by the wrong tag.
		{`{% if x %}a{% endfor %}`, "closed by"},
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
