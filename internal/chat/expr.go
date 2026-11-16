package chat

import (
	"fmt"
	"strconv"
	"strings"
)

// expr is a parsed template expression.
type expr interface{}

// litExpr is a constant: a string or a number, already decoded.
type litExpr struct{ val string }

// varExpr reads a variable from scope.
type varExpr struct{ name string }

// attrExpr reads a field from a map or a field of the loop variable.
type attrExpr struct {
	base expr
	name string
}

// indexExpr reads a sequence element by integer index.
type indexExpr struct {
	base  expr
	index expr
}

// concatExpr joins its operands, converting each to text.
type concatExpr struct{ parts []expr }

// filterExpr applies a named filter to its operand.
type filterExpr struct {
	// operand is what preceded the '|'.
	operand expr
	name    string
	arg     expr
}

// definedExpr reports whether a name is bound, which templates use to test optional
// inputs like tools.
type definedExpr struct{ name string }

// attrDefinedExpr reports whether a mapping has a field, without failing when it does
// not. Templates use it to branch on an optional member.
type attrDefinedExpr struct {
	base expr
	name string
}

// notExpr negates a condition.
type notExpr struct{ inner expr }

// binaryExpr is a comparison, boolean, or arithmetic operator.
type binaryExpr struct {
	op          string
	left, right expr
}

// arithExpr is integer arithmetic, which templates use to index relative to the loop
// position ("messages[loop.index0 - 1]").
type arithExpr struct {
	op          string
	left, right expr
}

// token is one lexical unit of an expression.
//
// The isString flag matters: a decoded literal like "abc" is indistinguishable from a
// bare name "abc", and treating a marker such as "<|im_start|>" as a name is what made
// every template with a special token in a string fail to parse.
type token struct {
	text     string
	isString bool
}

// supportedFilters is the filter set this package implements.
//
// It is checked at parse time so a template using an unimplemented filter fails to
// compile. Discovering it at render time would mean the caller had already committed to
// a prompt that cannot be finished.
var supportedFilters = map[string]bool{
	"trim":   true,
	"tojson": true,
	"length": true,
}

// exprParser parses one expression from a token stream.
type exprParser struct {
	toks []token
	pos  int
}

// parseExpr parses a complete expression, rejecting trailing input.
func parseExpr(src string) (expr, error) {
	toks, err := lexExpr(src)
	if err != nil {
		return nil, err
	}
	p := &exprParser{toks: toks}
	e, err := p.parseOr()
	if err != nil {
		return nil, err
	}
	if p.pos != len(p.toks) {
		return nil, fmt.Errorf("chat: unexpected %q after expression %q", p.toks[p.pos].text, src)
	}
	return e, nil
}

// lexExpr splits an expression into tokens, keeping quoted strings whole.
//
// Strings are decoded here rather than in the parser because a template's literals
// carry escapes ("\n" is a newline) and a chat format depends on them: a marker split
// across two lines is a different prompt.
func lexExpr(src string) ([]token, error) {
	var toks []token
	for i := 0; i < len(src); {
		c := src[i]
		switch {
		case c == ' ' || c == '\t' || c == '\r' || c == '\n':
			i++
		case c == '\'' || c == '"':
			s, n, err := lexString(src[i:])
			if err != nil {
				return nil, err
			}
			toks = append(toks, token{text: s, isString: true})
			i += n
		case c == '=' || c == '!' || c == '<' || c == '>':
			// Two-character comparisons first, so "==" is not read as two '='.
			if i+1 < len(src) && src[i+1] == '=' {
				toks = append(toks, token{text: src[i : i+2]})
				i += 2
				continue
			}
			if c == '=' {
				return nil, fmt.Errorf("chat: '=' is not a comparison; use '=='")
			}
			toks = append(toks, token{text: string(c)})
			i++
		case c == '|' || c == '(' || c == ')' || c == '[' || c == ']' ||
			c == '+' || c == '-' || c == '.' || c == ',':
			toks = append(toks, token{text: string(c)})
			i++
		default:
			start := i
			for i < len(src) && isExprWordByte(src[i]) {
				i++
			}
			if start == i {
				return nil, fmt.Errorf("chat: unexpected %q in expression %q", string(c), src)
			}
			toks = append(toks, token{text: src[start:i]})
		}
	}
	if len(toks) == 0 {
		return nil, fmt.Errorf("chat: empty expression")
	}
	return toks, nil
}

// isExprWordByte reports whether c can be part of a bare word: a name, a number, or a
// keyword. Booleans and operators spelled as words go through here too, which is why it
// accepts letters, digits, and underscore.
func isExprWordByte(c byte) bool {
	return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9'
}

// lexString reads one quoted literal and returns its decoded value and byte length.
func lexString(src string) (string, int, error) {
	quote := src[0]
	var b strings.Builder
	for i := 1; i < len(src); i++ {
		switch src[i] {
		case quote:
			return b.String(), i + 1, nil
		case '\\':
			if i+1 >= len(src) {
				return "", 0, fmt.Errorf("chat: string ends with a backslash")
			}
			i++
			switch src[i] {
			case 'n':
				b.WriteByte('\n')
			case 't':
				b.WriteByte('\t')
			case 'r':
				b.WriteByte('\r')
			case '\\':
				b.WriteByte('\\')
			case '\'':
				b.WriteByte('\'')
			case '"':
				b.WriteByte('"')
			default:
				// An unknown escape keeps the backslash, which is what Jinja does and
				// what a template containing a Windows path needs.
				b.WriteByte('\\')
				b.WriteByte(src[i])
			}
		default:
			b.WriteByte(src[i])
		}
	}
	return "", 0, fmt.Errorf("chat: unterminated string literal")
}

// peek returns the current token, or "" at end of input.
func (p *exprParser) peek() string {
	if p.pos >= len(p.toks) {
		return ""
	}
	return p.toks[p.pos].text
}

// peekToken returns the current token and whether it is a string literal.
func (p *exprParser) peekToken() (token, bool) {
	if p.pos >= len(p.toks) {
		return token{}, false
	}
	return p.toks[p.pos], true
}

// accept consumes the current token when it matches, reporting whether it did.
func (p *exprParser) accept(tok string) bool {
	if p.peek() != tok {
		return false
	}
	p.pos++
	return true
}

// parseOr handles "or", the lowest-precedence operator.
func (p *exprParser) parseOr() (expr, error) {
	left, err := p.parseAnd()
	if err != nil {
		return nil, err
	}
	for p.accept("or") {
		right, err := p.parseAnd()
		if err != nil {
			return nil, err
		}
		left = binaryExpr{op: "or", left: left, right: right}
	}
	return left, nil
}

// parseAnd handles "and".
func (p *exprParser) parseAnd() (expr, error) {
	left, err := p.parseNot()
	if err != nil {
		return nil, err
	}
	for p.accept("and") {
		right, err := p.parseNot()
		if err != nil {
			return nil, err
		}
		left = binaryExpr{op: "and", left: left, right: right}
	}
	return left, nil
}

// parseNot handles a leading "not".
//
// It binds looser than the comparison layer, so "not x is defined" is "not (x is
// defined)" -- which is what templates write and what Jinja means.
func (p *exprParser) parseNot() (expr, error) {
	if p.accept("not") {
		inner, err := p.parseNot()
		if err != nil {
			return nil, err
		}
		return notExpr{inner: inner}, nil
	}
	return p.parseCompare()
}

// parseCompare handles ==, !=, and "in".
func (p *exprParser) parseCompare() (expr, error) {
	left, err := p.parseConcat()
	if err != nil {
		return nil, err
	}
	for {
		op := p.peek()
		switch op {
		case "==", "!=":
			p.pos++
			right, err := p.parseConcat()
			if err != nil {
				return nil, err
			}
			left = binaryExpr{op: op, left: left, right: right}
		case "in":
			p.pos++
			right, err := p.parseConcat()
			if err != nil {
				return nil, err
			}
			left = binaryExpr{op: "in", left: left, right: right}
		case "is":
			p.pos++
			if !p.accept("defined") {
				return nil, fmt.Errorf("chat: only 'is defined' is supported after 'is'")
			}
			// "x is defined" and "x.y is defined" are both used: the first asks
			// whether a name is bound, the second whether a mapping has a key.
			switch v := left.(type) {
			case varExpr:
				left = definedExpr(v)
			case attrExpr:
				left = attrDefinedExpr(v)
			default:
				return nil, fmt.Errorf("chat: 'is defined' needs a variable or field name")
			}
		default:
			return left, nil
		}
	}
}

// parseConcat handles + between operands, folding an integer addition into arithmetic
// and everything else into string concatenation.
//
// The template language does not distinguish the two at parse time: "a" + "b" concatenates
// and 1 + 2 adds, and which one applies is decided from the operands at render time. The
// parser therefore builds a single add node and the evaluator picks the meaning.
func (p *exprParser) parseConcat() (expr, error) {
	first, err := p.parseArith()
	if err != nil {
		return nil, err
	}
	parts := []expr{first}
	for p.accept("+") {
		next, err := p.parseArith()
		if err != nil {
			return nil, err
		}
		parts = append(parts, next)
	}
	if len(parts) == 1 {
		return parts[0], nil
	}
	return concatExpr{parts: parts}, nil
}

// parseArith handles - between operands.
func (p *exprParser) parseArith() (expr, error) {
	left, err := p.parsePostfix()
	if err != nil {
		return nil, err
	}
	for p.accept("-") {
		right, err := p.parsePostfix()
		if err != nil {
			return nil, err
		}
		left = arithExpr{op: "-", left: left, right: right}
	}
	return left, nil
}

// parsePostfix handles attribute access, indexing, and filters after a primary.
func (p *exprParser) parsePostfix() (expr, error) {
	e, err := p.parsePrimary()
	if err != nil {
		return nil, err
	}
	for {
		switch {
		case p.accept("."):
			name := p.peek()
			if !isIdentifier(name) {
				return nil, fmt.Errorf("chat: expected a field name after '.'")
			}
			p.pos++
			e = attrExpr{base: e, name: name}
		case p.accept("["):
			idx, err := p.parseOr()
			if err != nil {
				return nil, err
			}
			if !p.accept("]") {
				return nil, fmt.Errorf("chat: missing ']' in index expression")
			}
			e = indexExpr{base: e, index: idx}
		case p.accept("|"):
			name := p.peek()
			if !isIdentifier(name) {
				return nil, fmt.Errorf("chat: expected a filter name after '|'")
			}
			p.pos++
			if !supportedFilters[name] {
				return nil, fmt.Errorf("chat: unsupported template filter %q", name)
			}
			fe := filterExpr{operand: e, name: name}
			// A filter may take one parenthesised argument: trim has none, tojson
			// has none, and selectattr-style filters are outside the subset.
			if p.accept("(") {
				arg, err := p.parseOr()
				if err != nil {
					return nil, err
				}
				if !p.accept(")") {
					return nil, fmt.Errorf("chat: missing ')' in filter arguments")
				}
				fe.arg = arg
			}
			e = fe
		default:
			return e, nil
		}
	}
}

// parsePrimary handles literals, names, and parenthesised expressions.
func (p *exprParser) parsePrimary() (expr, error) {
	tok := p.peek()
	if tok == "" {
		return nil, fmt.Errorf("chat: expression ends early")
	}
	if tok == "(" {
		p.pos++
		inner, err := p.parseOr()
		if err != nil {
			return nil, err
		}
		if !p.accept(")") {
			return nil, fmt.Errorf("chat: missing ')'")
		}
		return inner, nil
	}
	cur, _ := p.peekToken()
	p.pos++
	if cur.isString {
		return litExpr{val: cur.text}, nil
	}
	if tok == "-" {
		// A leading minus belongs to the number that follows, so it is folded here
		// rather than becoming a unary node. Only integers can be negated in this
		// subset, which is all a template index needs.
		next, ok := p.peekToken()
		if !ok {
			return nil, fmt.Errorf("chat: expression ends after '-'")
		}
		p.pos++
		return litExpr{val: "-" + next.text}, nil
	}
	switch tok {
	case "true", "True":
		return litExpr{val: "true"}, nil
	case "false", "False":
		return litExpr{val: "false"}, nil
	case "none", "None":
		return litExpr{val: ""}, nil
	}
	if n, err := strconv.Atoi(tok); err == nil {
		return litExpr{val: strconv.Itoa(n)}, nil
	}
	if !isIdentifier(tok) {
		return nil, fmt.Errorf("chat: unexpected %q in expression", tok)
	}
	return varExpr{name: tok}, nil
}
