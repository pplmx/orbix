package chat

import (
	"fmt"
	"strconv"
	"strings"
)

// node is one executable element of a template.
type node interface{}

// textNode emits literal text.
type textNode struct{ text string }

// exprNode emits an expression's value.
type exprNode struct{ expr expr }

// setNode assigns a variable for the rest of the enclosing scope.
//
// attr is set for the `{% set ns.field = ... %}` form, which assigns through a namespace rather
// than binding a new name. Qwen3's template uses it to carry loop state (`multi_step_tool`,
// `last_query_index`) out of a for body, which a plain `set` inside the body cannot do because
// that body is its own frame.
type setNode struct {
	name string
	attr string
	expr expr
}

// ifNode is one branch chain. Only the first branch whose condition holds runs.
type ifNode struct {
	branches []ifBranch
	// elseBody runs when no condition held. A template with an else is
	// distinguishable from one without: the latter emits nothing there.
	elseBody []node
	hasElse  bool
}

// ifBranch is one condition and its body.
type ifBranch struct {
	cond expr
	body []node
}

// forNode iterates a sequence bound to a name, exposing the loop variable.
type forNode struct {
	// names holds the bound names: one for the ordinary form, two for a tuple unpacking like
	// `{% for key, val in tool.items() %}`.
	names []string
	iter  expr
	// filter is the loop's `if condition`, evaluated per item; nil when the header has none.
	filter expr
	body   []node
}

// parser turns a piece list into a node tree.
//
// It is recursive descent over the statement pieces: {% if %} opens a body that runs
// until {% elif %}, {% else %}, or {% endif %}, and {% for %} likewise until
// {% endfor %}. Nesting comes from the recursion rather than from an explicit stack,
// which keeps the error messages tied to the construct that failed.
type parser struct {
	pieces []piece
	pos    int
}

// parse builds the tree for a template.
func parse(pieces []piece) ([]node, error) {
	p := &parser{pieces: pieces}
	nodes, term, err := p.parseBody()
	if err != nil {
		return nil, err
	}
	if term.keyword != "" {
		return nil, fmt.Errorf("chat: unexpected {%% %s %%} with no open block, at offset %d",
			term.keyword, term.offset)
	}
	return nodes, nil
}

// term is a closing statement: its keyword and its argument.
//
// The argument travels with the keyword because the caller needs it -- "elif y" is an
// expression "y" plus a terminator "elif", and re-splitting the statement at the call
// site would duplicate the split rules.
type term struct {
	keyword string
	// offset is where the closing tag begins, so an error about it can point at the line. The
	// enclosing block reports these, and a template with several if blocks would otherwise
	// name only the keyword.
	offset int
	arg    string
}

// parseBody parses nodes until a statement that closes the current block, which it
// reports as a terminator without consuming it. An empty keyword means end of input.
func (p *parser) parseBody() (nodes []node, t term, err error) {
	for p.pos < len(p.pieces) {
		pc := p.pieces[p.pos]
		switch pc.kind {
		case pieceText:
			p.pos++
			if pc.text != "" {
				nodes = append(nodes, textNode{text: pc.text})
			}
		case pieceExpr:
			p.pos++
			e, err := parseExpr(pc.text)
			if err != nil {
				return nil, term{}, err
			}
			nodes = append(nodes, exprNode{expr: e})
		case pieceStmt:
			keyword, rest, err := splitStatement(pc.text)
			if err != nil {
				return nil, term{}, err
			}
			switch keyword {
			case "if":
				p.pos++
				n, err := p.parseIf(rest, pc.offset)
				if err != nil {
					return nil, term{}, err
				}
				nodes = append(nodes, n)
			case "for":
				p.pos++
				n, err := p.parseFor(rest, pc.offset)
				if err != nil {
					return nil, term{}, err
				}
				nodes = append(nodes, n)
			case "set":
				p.pos++
				n, err := parseSet(rest)
				if err != nil {
					return nil, term{}, err
				}
				nodes = append(nodes, n)
			case "elif", "else", "endif", "endfor":
				// The caller consumes this; leaving pos on it is what lets the
				// enclosing block decide what to do.
				return nodes, term{keyword: keyword, arg: rest, offset: pc.offset}, nil
			default:
				return nil, term{}, fmt.Errorf("chat: unsupported template statement %q at offset %d",
					keyword, pc.offset)
			}
		}
	}
	return nodes, term{}, nil
}

// parseIf parses an if chain, having consumed the {% if %}.
func (p *parser) parseIf(condSrc string, start int) (node, error) {
	cond, err := parseExpr(condSrc)
	if err != nil {
		return nil, err
	}
	out := ifNode{}
	body, t, err := p.parseBody()
	if err != nil {
		return nil, err
	}
	out.branches = append(out.branches, ifBranch{cond: cond, body: body})

	for {
		if t.keyword == "" {
			return nil, fmt.Errorf("chat: {%% if %%} opened at offset %d is never closed", start)
		}
		p.pos++ // consume the terminator now that its keyword is known
		switch t.keyword {
		case "elif":
			c, err := parseExpr(t.arg)
			if err != nil {
				return nil, err
			}
			b, next, err := p.parseBody()
			if err != nil {
				return nil, err
			}
			out.branches = append(out.branches, ifBranch{cond: c, body: b})
			t = next
		case "else":
			b, next, err := p.parseBody()
			if err != nil {
				return nil, err
			}
			if next.keyword != "endif" {
				return nil, fmt.Errorf("chat: {%% else %%} must be followed by {%% endif %%}")
			}
			p.pos++ // consume {% endif %}
			out.elseBody, out.hasElse = b, true
			return out, nil
		case "endif":
			return out, nil
		default:
			return nil, fmt.Errorf("chat: {%% if %%} opened at offset %d closed by {%% %s %%} at offset %d",
				start, t.keyword, t.offset)
		}
	}
}

// parseFor parses a for loop, having consumed the {% for %}.
// forNames splits what precedes " in " into the loop's bound names.
//
// Jinja2 binds either one name or a tuple of them, and the tuple form unpacks each item --
// `{% for key, val in tool.items() %}`, which Mistral's templates use to walk a tool's arguments.
// The names are returned separately rather than as a slice so the single-name case, which is nearly
// all of them, stays the obvious path.
func forNames(spec string) (names []string, err error) {
	for _, part := range strings.Split(spec, ",") {
		name := strings.TrimSpace(part)
		if !isIdentifier(name) {
			return nil, fmt.Errorf("chat: {%% for %%} binds an invalid name %q", name)
		}
		names = append(names, name)
	}
	if len(names) > 2 {
		return nil, fmt.Errorf("chat: {%% for %%} binds %d names; at most two are supported, which "+
			"is what a mapping's items unpack to", len(names))
	}
	return names, nil
}

// splitForIf separates a for header's optional trailing filter.
//
// `{% for x in xs if cond %}` is a loop filter, not an inline conditional: the condition decides
// which items are iterated. Handing the whole header to the expression parser made it read the
// `if` as the ternary form, which needs an `else`, and refused templates that are valid Jinja2.
//
// The split is done on tokens rather than on the string, because a filter expression can contain
// the word -- a string literal, or a nested comparison -- and searching for " if " would cut in the
// wrong place.
func splitForIf(header string) (loop, cond string, err error) {
	toks, err := lexExpr(header)
	if err != nil {
		return "", "", err
	}
	depth := 0
	for i, t := range toks {
		switch t.text {
		case "(", "[", "{":
			depth++
		case ")", "]", "}":
			depth--
		case "if":
			// The first top-level `if` is the filter. A parenthesised or subscripted one belongs
			// to an expression being filtered.
			if depth == 0 {
				if i == len(toks)-1 {
					// A filter with no condition is a syntax error, not the absence of a filter.
					// Jinja2 refuses it ("unexpected 'end of statement block'"); rendering the
					// loop unfiltered instead made a broken template work, which is the one
					// outcome that hides itself.
					return "", "", fmt.Errorf("chat: {%% for %%} ends with a bare 'if'; it needs " +
						"a condition")
				}
				return joinTokens(toks[:i]), joinTokens(toks[i+1:]), nil
			}
		}
	}
	return header, "", nil
}

// joinTokens rebuilds source text from tokens for the sub-parsers, which take strings.
//
// A string literal has to be re-quoted: the token carries its value with quotes stripped, so
// emitting the bare value turned `k != "b"` into `k != b`, where `b` is a name rather than a
// string. Measured before the fix: `{% for k, v in pairs if k != "b" %}` rendered nothing, because
// the filter compared each key against an unbound name and every item was dropped. Double quotes
// and escaped backslashes are enough to round-trip any value the lexer produced.
func joinTokens(toks []token) string {
	parts := make([]string, len(toks))
	for i, t := range toks {
		if t.isString {
			parts[i] = strconv.Quote(t.text)
			continue
		}
		parts[i] = t.text
	}
	return strings.Join(parts, " ")
}

func (p *parser) parseFor(header string, start int) (node, error) {
	// The header is "names in expression", optionally followed by "if condition".
	spec, rest, ok := strings.Cut(header, " in ")
	if !ok {
		return nil, fmt.Errorf("chat: {%% for %s %%} has no 'in' clause", header)
	}
	names, err := forNames(strings.TrimSpace(spec))
	if err != nil {
		return nil, err
	}
	iterSrc, filterSrc, err := splitForIf(strings.TrimSpace(rest))
	if err != nil {
		return nil, err
	}
	iter, err := parseExpr(strings.TrimSpace(iterSrc))
	if err != nil {
		return nil, err
	}
	var filter expr
	if strings.TrimSpace(filterSrc) != "" {
		if filter, err = parseExpr(strings.TrimSpace(filterSrc)); err != nil {
			return nil, err
		}
	}
	body, t, err := p.parseBody()
	if err != nil {
		return nil, err
	}
	if t.keyword != "endfor" {
		if t.keyword == "" {
			return nil, fmt.Errorf("chat: {%% for %%} opened at offset %d is never closed", start)
		}
		return nil, fmt.Errorf("chat: {%% for %%} opened at offset %d closed by {%% %s %%} at offset %d",
			start, t.keyword, t.offset)
	}
	p.pos++
	return forNode{names: names, iter: iter, filter: filter, body: body}, nil
}

// parseSet parses a {% set %} assignment.
func parseSet(src string) (node, error) {
	name, rest, ok := strings.Cut(src, "=")
	if !ok {
		return nil, fmt.Errorf("chat: {%% set %s %%} has no '='", src)
	}
	name = strings.TrimSpace(name)
	// The dotted form assigns through a namespace. Only one level is accepted, because that is
	// what the namespace construct produces: a flat bag of fields, with no nesting to walk.
	base, attr, dotted := strings.Cut(name, ".")
	if dotted {
		if !isIdentifier(base) || !isIdentifier(attr) {
			return nil, fmt.Errorf("chat: {%% set %%} assigns to an invalid name %q", name)
		}
		e, err := parseExpr(strings.TrimSpace(rest))
		if err != nil {
			return nil, err
		}
		return setNode{name: base, attr: attr, expr: e}, nil
	}
	if !isIdentifier(name) {
		return nil, fmt.Errorf("chat: {%% set %%} assigns to an invalid name %q", name)
	}
	e, err := parseExpr(strings.TrimSpace(rest))
	if err != nil {
		return nil, err
	}
	return setNode{name: name, expr: e}, nil
}

// splitStatement separates a statement's keyword from its argument.
func splitStatement(src string) (keyword, rest string, err error) {
	src = strings.TrimSpace(src)
	if src == "" {
		return "", "", fmt.Errorf("chat: empty template statement")
	}
	keyword, rest, _ = strings.Cut(src, " ")
	return keyword, strings.TrimSpace(rest), nil
}

// isIdentifier reports whether s is a valid variable name.
func isIdentifier(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		switch {
		case r == '_', r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
			if i == 0 {
				return false
			}
		default:
			return false
		}
	}
	return true
}
