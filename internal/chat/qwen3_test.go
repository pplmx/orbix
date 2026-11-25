package chat

import "testing"

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
