package chat

import (
	"fmt"
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
	varName string
	iter    expr
	body    []node
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
func (p *parser) parseFor(header string, start int) (node, error) {
	// The header is "name in expression".
	name, rest, ok := strings.Cut(header, " in ")
	if !ok {
		return nil, fmt.Errorf("chat: {%% for %s %%} has no 'in' clause", header)
	}
	name = strings.TrimSpace(name)
	if !isIdentifier(name) {
		return nil, fmt.Errorf("chat: {%% for %%} binds an invalid name %q", name)
	}
	iter, err := parseExpr(strings.TrimSpace(rest))
	if err != nil {
		return nil, err
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
	return forNode{varName: name, iter: iter, body: body}, nil
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
