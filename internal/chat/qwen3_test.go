package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestNamespaceCarriesStateOutOfALoop covers the namespace construct.
//
// It is the one piece of mutable state in Jinja's expression language, and Qwen3's chat template
// uses it to remember the last user turn's index from inside a for body. A plain `{% set %}`
// there binds in the body's frame and disappears with it, which is exactly why the template
// reaches for a namespace instead.
//
// Measured before this was supported: every Qwen3 conversation fell back to the neutral
// transcript, and a two-turn probe answered "What is my name?" correctly and then went on
// inventing further turns.
func TestNamespaceCarriesStateOutOfALoop(t *testing.T) {
	cases := []struct{ name, src, want string }{
		{
			name: "fields read back",
			src:  `{% set ns = namespace(a=1, b=2) %}{{ ns.a }}-{{ ns.b }}`,
			want: "1-2",
		},
		{
			name: "a field assigned after construction",
			src:  `{% set ns = namespace(x=1) %}{% set ns.x = 9 %}{{ ns.x }}`,
			want: "9",
		},
		{
			name: "a field assigned inside a for body survives it",
			src: `{% set ns = namespace(last=-1) %}` +
				`{% for m in messages %}{% set ns.last = loop.index0 %}{% endfor %}{{ ns.last }}`,
			want: "2",
		},
		{
			// Printing the bag is a template bug, and this package refuses it rather than
			// emitting Go's map formatting. Jinja2 prints a repr that no template wants.
			name: "printing the bag is refused",
			src:  `{% set ns = namespace(x=1) %}[{{ ns }}]`,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			msgs := []Message{{Role: "user", Content: "a"}, {Role: "user", Content: "b"}, {Role: "user", Content: "c"}}
			got, err := tpl.Render(msgs, Inputs{})
			if c.want == "" && c.name == "printing the bag is refused" {
				if err == nil {
					t.Fatalf("printing a namespace rendered %q; it should be refused", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("Render: %v", err)
			}
			if got != c.want {
				t.Errorf("rendered %q, want %q", got, c.want)
			}
		})
	}
}

// TestNamespaceTakesKeywordArgumentsOnly covers the refusal for a positional argument.
//
// namespace() is built from keyword arguments because a field with no name is not a field --
// the construct is a named bag, and `namespace(1)` cannot say which name the 1 belongs to.
// The refusal had no test: clearing the args instead of refusing left the package green.
func TestNamespaceTakesKeywordArgumentsOnly(t *testing.T) {
	for _, src := range []string{
		`{% set ns = namespace('x') %}{{ ns }}`,
		`{% set ns = namespace(1, 2) %}{{ ns }}`,
	} {
		tpl, err := Parse(src)
		if err != nil {
			t.Fatalf("Parse(%q): %v", src, err)
		}
		_, err = tpl.Render(nil, Inputs{})
		if err == nil {
			t.Errorf("%q accepted a positional argument, which cannot name a field", src)
			continue
		}
		if !strings.Contains(err.Error(), "keyword arguments only") {
			t.Errorf("%q: error %q does not say only keywords are accepted", src, err)
		}
	}
}

// TestNamespaceRefusesAssignmentThroughSomethingElse keeps the dotted set from silently
// creating a name nobody reads.
//
// `{% set x.y = 1 %}` on a plain string would otherwise "succeed" and produce a prompt that
// looks right while the assignment went nowhere -- the class of failure this package refuses
// rather than rounds off.
func TestNamespaceRefusesAssignmentThroughSomethingElse(t *testing.T) {
	for _, src := range []string{
		`{% set s = 'text' %}{% set s.field = 1 %}{{ s.field }}`,
		`{% set undefined_ns.field = 1 %}{{ undefined_ns.field }}`,
	} {
		tpl, err := Parse(src)
		if err != nil {
			continue // refusing at parse time is also fine
		}
		if _, err := tpl.Render(nil, Inputs{}); err == nil {
			t.Errorf("%q rendered without an error, so the assignment went nowhere silently", src)
		}
	}
}

// TestStringMethodsMatchPython pins the four string methods Qwen3's template uses.
//
// The argument is a character set, not a substring, which is Python's rule: `'xxhixx'.strip('x')`
// is "hi" because every leading and trailing x goes, while `strings.Trim(s, "xx")` happens to
// agree here and would not for a two-character set. The cases below include one where a
// substring reading would differ, so a future rewrite to strings.TrimPrefix is caught.
func TestStringMethodsMatchPython(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ '  hi  '.strip() }}`, "hi"},
		{`{{ 'xxhixx'.strip('x') }}`, "hi"},
		{`{{ 'xyhixy'.strip('xy') }}`, "hi"},
		{`{{ '  hi  '.lstrip() }}`, "hi  "},
		{`{{ '  hi  '.rstrip() }}`, "  hi"},
		{`{{ "ab" + "ab".strip("ab") }}`, "ab"},
		{`{{ 'a,b,c'.split(',')|length }}`, "3"},
		{`{{ 'a,b,c'.split(',')|first }}`, "a"},
		{`{{ 'a,b,c'.split(',')|last }}`, "c"},
		{`{{ 'no sep'.split('|')|length }}`, "1"},
		{`{{ 'a b  c'.split()|length }}`, "3"},
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
			t.Errorf("%q => %q, want %q", c.src, got, c.want)
		}
	}
}

// TestStringMethodRefusals covers the error paths of the methods above.
//
// TestStringMethodsMatchPython exercises only the successes. Five refusals in evalStringMethod
// had no test at all: calling a method on a boolean, more than one argument to strip/split,
// an empty strip set, and an empty split separator. Replacing each return with a value left
// the whole chat suite green, so the messages a template author would see were never read.
func TestStringMethodRefusals(t *testing.T) {
	cases := []struct{ src, wantSubstr string }{
		// A boolean has no text to strip; saying so is better than rendering "True".
		{`{% set b = true %}{{ b.strip() }}`, "called on a boolean"},
		{`{{ 'x'.strip('a', 'b') }}`, "takes at most one argument"},
		// Python's str.strip("") raises ValueError; this says the same thing in words.
		{`{{ 'x'.strip('') }}`, "removes nothing"},
		{`{{ 'x'.split('') }}`, "empty separator"},
		{`{{ 'x'.split(',', 1) }}`, "split() takes at most one argument"},
	}
	for _, c := range cases {
		tpl, err := Parse(c.src)
		if err != nil {
			t.Errorf("Parse(%q): %v", c.src, err)
			continue
		}
		out, err := tpl.Render(nil, Inputs{})
		if err == nil {
			t.Errorf("%q rendered %q, want a refusal", c.src, out)
			continue
		}
		if !strings.Contains(err.Error(), c.wantSubstr) {
			t.Errorf("%q: error %q does not say %q", c.src, err, c.wantSubstr)
		}
		// Rendering nothing alongside an error is the quiet half-render to avoid.
		if out != "" {
			t.Errorf("%q produced %q alongside its error", c.src, out)
		}
	}
}

// TestOrderingComparisons covers <, >, <= and >=, which were missing entirely.
//
// Qwen3's template writes `{% if start_pos < 0 %}`. Before this, the parser stopped at the "<"
// and reported "unexpected "<" after expression", so the whole template fell back. Numbers
// compare numerically and text lexically, which is Python 3's rule.
func TestOrderingComparisons(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{% if 1 < 2 %}y{% else %}n{% endif %}`, "y"},
		{`{% if 2 < 2 %}y{% else %}n{% endif %}`, "n"},
		{`{% if 2 <= 2 %}y{% else %}n{% endif %}`, "y"},
		{`{% if 3 > 2 %}y{% else %}n{% endif %}`, "y"},
		{`{% if 2 >= 3 %}y{% else %}n{% endif %}`, "n"},
		// Numeric, not lexical: "10" sorts before "9" as text.
		{`{% if 10 > 9 %}y{% else %}n{% endif %}`, "y"},
		{`{% if -1 < 0 %}y{% else %}n{% endif %}`, "y"},
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
			t.Errorf("%q => %q, want %q", c.src, got, c.want)
		}
	}
}

// TestIsTrueAndIsFalseTests covers the two is-tests Qwen3 uses.
//
// They ask whether a value *is* the boolean, not whether it is truthy: Jinja2 answers False for
// `1 is true`. Qwen3 applies them to a namespace field it assigned true to, where the
// distinction does not bite -- but an equality test would be wrong the first time a template
// applied it to a number, so the distinction is pinned here.
func TestIsTrueAndIsFalseTests(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{% if true is true %}y{% else %}n{% endif %}`, "y"},
		{`{% if false is true %}y{% else %}n{% endif %}`, "n"},
		{`{% if false is false %}y{% else %}n{% endif %}`, "y"},
		{`{% if true is false %}y{% else %}n{% endif %}`, "n"},
		{`{% if 1 is true %}y{% else %}n{% endif %}`, "n"},
		{`{% if 0 is false %}y{% else %}n{% endif %}`, "n"},
		{`{% if true is not false %}y{% else %}n{% endif %}`, "y"},
		{`{% if false is not true %}y{% else %}n{% endif %}`, "y"},
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
			t.Errorf("%q => %q, want %q", c.src, got, c.want)
		}
	}
}

// TestStringSlicing covers the [:n] form, which Qwen3 uses to take the payload before a marker.
//
// The [n:] form already existed for Llama 3's messages[1:]. Both clamp out-of-range bounds
// rather than refusing, because a template that guards the length should not also have to
// guard the slice -- Python clamps too, which is what these templates were written against.
func TestStringSlicing(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ 'hello'[:3] }}`, "hel"},
		{`{{ 'hello'[2:] }}`, "llo"},
		{`{{ 'hello'[1:3] }}`, "el"},
		{`{{ 'hello'[:99] }}`, "hello"},
		{`{{ 'hello'[9:] }}`, ""},
		{`{{ 'hello'[-2:] }}`, "lo"},
		{`{{ ''[:1] }}`, ""},
		// Runes, not bytes: an offset in the middle of a multi-byte character must not split it.
		{`{{ '中文'[:1] }}`, "中"},
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
			t.Errorf("%q => %q, want %q", c.src, got, c.want)
		}
	}
}

// TestJoinIsRefusedAtTheRightStage pins where Jinja's join stops working here.
//
// The package never supported join, but it used to look as though it did: evalStringMethod
// carried a "join" case, and no test or document mentioned it. That case could not run. Its
// argument is the separator and its receiver has to be a sequence, while the value a method
// call delivers is always text, so every spelling failed with "join() needs a sequence" --
// including one over a slice and one bound through {% set %}.
//
// The two refusals happen at different stages for a reason worth keeping straight:
// `list | join` is a filter, and an unsupported filter name is a parse error (so the caller
// falls back before committing to a prompt); `x.join(...)` is a method call, which parses
// anywhere and fails when evaluated. This asserts both, and asserts the stage, because a
// change that moved either one would be a change to the fallback contract.
func TestJoinIsRefusedAtTheRightStage(t *testing.T) {
	// As a filter: refused at parse time, naming the filter.
	if _, err := Parse(`{{ messages | join('-') }}`); err == nil {
		t.Fatal("parsing `messages | join(...)` succeeded; an unsupported filter must fail at parse time")
	} else if !strings.Contains(err.Error(), "unsupported template filter") {
		t.Errorf("the parse error was %q; it should name the unsupported filter, so a reader "+
			"can tell it from a syntax error", err)
	}

	// As a method: parses, then fails at render time.
	tpl, err := Parse(`{{ messages.join('-') }}`)
	if err != nil {
		t.Fatalf("Parse(`messages.join(...)`): %v -- a method call parses anywhere, which is "+
			"why the unimplemented ones surface at render time", err)
	}
	_, err = tpl.Render(nil, Inputs{})
	if err == nil {
		t.Fatal("rendering `messages.join(...)` succeeded; nothing implements it")
	}
	if !strings.Contains(err.Error(), "not implemented") {
		t.Errorf("the render error was %q; it should say the method is not implemented", err)
	}
}

// TestQwen3ReleasedTemplateRenders covers the checkpoint's own template, not a constructed one.
//
// Every other Qwen3 test in this file exercises a single construct against a string typed by
// hand. That is how the constructs were added, and it left a gap that mattered: /models/Qwen3-0.6B
// ships a 4168-character template that uses `messages[::-1]` and `.startswith()`, neither of which
// existed, so it was refused outright and every conversation with that checkpoint silently used
// the neutral transcript. The construct tests passed the whole time. The difference is that a
// construct test proves a feature works; only the whole template proves the feature is what the
// model needed.
//
// They were found by pointing the package at the real file and reading the error, which is what
// this test now does on every run. The expected outputs are the whole prompt, byte for byte,
// because a chat template's spacing is part of the format and a difference of one newline moves
// the model off its distribution.
func TestQwen3ReleasedTemplateRenders(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "qwen3-0.6b.jinja"))
	if err != nil {
		t.Fatalf("read the Qwen3 asset: %v", err)
	}
	tpl, err := Parse(string(raw))
	if err != nil {
		t.Fatalf("the released Qwen3 template does not parse: %v\n"+
			"This is the failure that matters for a checkpoint: the caller falls back to the "+
			"neutral transcript, so the model is served a prompt it was not trained on.", err)
	}

	conversation := []Message{
		{Role: "system", Content: "You are terse."},
		{Role: "user", Content: "What is 2+2?"},
		{Role: "assistant", Content: "4"},
		{Role: "user", Content: "And times 3?"},
	}
	// Verified byte-identical against Jinja2 3.1.6 rendering the same template and messages.
	want := "<|im_start|>system\nYou are terse.<|im_end|>\n" +
		"<|im_start|>user\nWhat is 2+2?<|im_end|>\n" +
		"<|im_start|>assistant\n4<|im_end|>\n" +
		"<|im_start|>user\nAnd times 3?<|im_end|>\n" +
		"<|im_start|>assistant\n"
	got, err := tpl.Render(conversation, Inputs{AddGenerationPrompt: true})
	if err != nil {
		t.Fatalf("rendering the released template: %v", err)
	}
	if got != want {
		t.Errorf("rendered\n%q\nwant\n%q", got, want)
	}

	// The tools branch, which is the reason the template is this long. Compared with the
	// parameters object written the way json.dumps writes it inside tojson -- the package
	// emits a caller's parameters verbatim rather than re-encoding them, so the input has to
	// be the text both sides would see.
	tools := []Tool{{Name: "get", Description: "fetch", Parameters: `{"type": "object"}`}}
	withTools, err := tpl.Render(conversation[:2], Inputs{AddGenerationPrompt: true, Tools: tools})
	if err != nil {
		t.Fatalf("rendering the tools branch: %v", err)
	}
	for _, want := range []string{
		"# Tools\n\nYou may call one or more functions",
		`{"function": {"description": "fetch", "name": "get", "parameters": {"type": "object"}}, "type": "function"}`,
		"<tool_call>",
		"<|im_start|>assistant\n",
	} {
		if !strings.Contains(withTools, want) {
			t.Errorf("the tools branch is missing %q; got:\n%s", want, withTools)
		}
	}

	// The reverse walk the template uses to find the last genuine user turn. If [::-1] rendered
	// empty or in order, the branch below would take the wrong arm and the prompt would still
	// look plausible -- the failure mode this whole package exists to avoid.
	reversed := render(t, `{% for m in messages[::-1] %}{{ m.role }} {% endfor %}`, conversation, Inputs{})
	if reversed != "user assistant user system " {
		t.Errorf("[::-1] over the conversation rendered %q, want the roles in reverse order", reversed)
	}
}

// TestSliceStep covers the third slice field, which Qwen3's template needs to walk a
// conversation backwards.
//
// The expectations are Python's, checked against Jinja2 3.1.6: a negative step reverses and its
// absent bounds mean the last element and "past the beginning", which is why the implementation
// cannot reuse the forward [start:end] arithmetic.
func TestSliceStep(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{% for m in messages[::-1] %}{{ m.role }}{% endfor %}`, "dcba"},
		{`{% for m in messages[::-2] %}{{ m.role }}{% endfor %}`, "db"},
		{`{% for m in messages[3:0:-1] %}{{ m.role }}{% endfor %}`, "dcb"},
		{`{% for m in messages[::2] %}{{ m.role }}{% endfor %}`, "ac"},
		{`{% for m in messages[:] %}{{ m.role }}{% endfor %}`, "abcd"},
		{`{% for m in messages[1:3] %}{{ m.role }}{% endfor %}`, "bc"},
		{`{% for m in messages[-2:] %}{{ m.role }}{% endfor %}`, "cd"},
		{`{% for m in messages[:-2] %}{{ m.role }}{% endfor %}`, "ab"},
		{`{{ 'abcdef'[::-1] }}`, "fedcba"},
		{`{{ 'abcdef'[::2] }}`, "ace"},
		{`{{ 'abcdef'[4:1:-1] }}`, "edc"},
		// Out-of-range bounds clamp rather than fail, as in Python.
		{`{{ 'abcdef'[-99:] }}`, "abcdef"},
		{`{{ 'abcdef'[:99] }}`, "abcdef"},
		{`{{ 'abcdef'[99:] }}`, ""},
		{`{{ 'abcdef'[:-99] }}`, ""},
	}
	msgs := []Message{{Role: "a"}, {Role: "b"}, {Role: "c"}, {Role: "d"}}
	for _, c := range cases {
		if got := render(t, c.src, msgs, Inputs{}); got != c.want {
			t.Errorf("%q => %q, want %q", c.src, got, c.want)
		}
	}

	// A zero step is Python's ValueError. Rendering an empty loop instead would be a prompt
	// with a silently missing section, which is the failure this package refuses to produce.
	tpl, err := Parse(`{% for m in messages[::0] %}{{ m.role }}{% endfor %}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if _, err := tpl.Render(msgs, Inputs{}); err == nil {
		t.Fatal("a zero slice step rendered; Python raises ValueError and this must refuse too")
	}

	// And a step that is not a number at all. strconv.Atoi's error used to be swallowed by
	// falling back to 1, so `xs[::'a']` silently meant `xs[::1]` -- the same shape of quiet
	// substitution the zero case is guarded against.
	for _, src := range []string{`{{ 'abcdef'[::'a'] }}`, `{% set s = 'b' %}{{ 'abcdef'[::s] }}`} {
		tpl, err := Parse(src)
		if err != nil {
			t.Fatalf("Parse(%q): %v", src, err)
		}
		_, err = tpl.Render(msgs, Inputs{})
		if err == nil {
			t.Errorf("%q rendered, want a refusal: a non-numeric step is not a stride of 1", src)
			continue
		}
		if !strings.Contains(err.Error(), "is not a number") {
			t.Errorf("%q: error %q does not say the step is not a number", src, err)
		}
	}
}

// TestStartswithEndswith covers the two predicates Qwen3's tool-response guard uses.
//
// They are here rather than in evalStringMethod's test because the reason they exist is a
// released template, and that is what the comment should point at: without them the guard
// `not (content.startswith('<tool_response>') and content.endswith('</tool_response>'))` fails
// to evaluate and the whole 4168-character template is refused.
func TestStartswithEndswith(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{% if 'abc'.startswith('ab') %}y{% else %}n{% endif %}`, "y"},
		{`{% if 'abc'.startswith('bc') %}y{% else %}n{% endif %}`, "n"},
		{`{% if 'abc'.endswith('bc') %}y{% else %}n{% endif %}`, "y"},
		{`{% if 'abc'.endswith('ab') %}y{% else %}n{% endif %}`, "n"},
		// The empty prefix is inside every string, which is Python's answer.
		{`{% if 'abc'.startswith('') %}y{% else %}n{% endif %}`, "y"},
		// The guard the real template writes, in both directions.
		{`{% if '<tool_response>x</tool_response>'.startswith('<tool_response>') and ` +
			`'<tool_response>x</tool_response>'.endswith('</tool_response>') %}wrapped{% endif %}`, "wrapped"},
		{`{% if 'plain'.startswith('<tool_response>') %}wrapped{% endif %}`, ""},
	}
	for _, c := range cases {
		if got := render(t, c.src, nil, Inputs{}); got != c.want {
			t.Errorf("%q => %q, want %q", c.src, got, c.want)
		}
	}
}
