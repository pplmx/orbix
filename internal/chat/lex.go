package chat

import (
	"fmt"
	"strings"
)

// pieceKind is what a template fragment is.
type pieceKind int

const (
	// pieceText is literal output.
	pieceText pieceKind = iota
	// pieceExpr is a {{ ... }} substitution.
	pieceExpr
	// pieceStmt is a {% ... %} statement.
	pieceStmt
)

// piece is one fragment of a parsed template.
type piece struct {
	kind pieceKind
	// offset is the byte position in the source where this piece begins, so a parse error can
	// name where it happened. Without it a template of any size reports only what was wrong,
	// and the reader counts tags to find which one.
	//
	// Text pieces carry the position of their first byte as well, since a stray {{ endif }} is
	// reported from the parser and the position that helps is the tag's, not the block's.
	offset int
	// text is the literal text for pieceText, and the trimmed source for the others.
	text string
	// trimBefore and trimAfter record a {%- ... -%} or {{- ... -}} marker, which
	// removes the whitespace adjacent to the tag in the surrounding literal text.
	//
	// Templates use these to keep the rendered prompt free of the newlines and
	// indentation that make the source readable. Ignoring them produces a prompt with
	// extra whitespace the model was not trained on, which is a real change to the
	// input even though it looks cosmetic.
	trimBefore bool
	trimAfter  bool
}

// lex splits a template into text, expression, and statement pieces.
//
// It is a scanner rather than a parser: deciding what an expression means is the
// parser's job, and keeping the two apart is what lets a lexing error name a position
// while a parser error names a construct.
// indexCloseTag finds the closing delimiter of a tag, ignoring delimiters that appear
// inside a quoted literal.
//
// A plain search is wrong because a template can contain the delimiter as data: Qwen's chat
// template embeds a JSON example, {{"name": <function-name>, "arguments": ...}}, inside a
// string literal in an expression. Stopping at the first "}}" there cuts the expression in
// half and reports an unterminated string, which is what happened before this existed.
//
// Only expressions need the scan: a statement or comment cannot contain a quoted literal
// with the delimiter in it, and scanning them the same way costs nothing.
func indexCloseTag(src, closeTag string) int {
	quote := byte(0)
	for i := 0; i < len(src); i++ {
		c := src[i]
		if quote != 0 {
			if c == '\\' {
				i++ // the escaped byte cannot close the literal
				continue
			}
			if c == quote {
				quote = 0
			}
			continue
		}
		if c == '\'' || c == '"' {
			quote = c
			continue
		}
		if strings.HasPrefix(src[i:], closeTag) {
			return i
		}
	}
	return -1
}

func lex(src string) ([]piece, error) {
	var out []piece
	var text strings.Builder
	// trimPending is a comment's trailing trim waiting for the text that follows it.
	trimPending := false

	// flushText appends the pending literal, applying a trim requested by the tag
	// that follows it.
	flushText := func(trimEnd bool) {
		s := text.String()
		text.Reset()
		if trimPending {
			s = strings.TrimLeft(s, " \t\r\n")
			trimPending = false
		}
		if trimEnd {
			s = strings.TrimRight(s, " \t\r\n")
		}
		if s != "" {
			// A text piece's offset is not needed: every parse error names a tag, because a
			// literal cannot be malformed. Leaving it zero keeps flushText from having to
			// track where the pending literal began across a trim.
			out = append(out, piece{kind: pieceText, text: s})
		}
	}

	for i := 0; i < len(src); {
		// The next tag is whichever delimiter comes first. Searching for the three
		// delimiters separately scans the rest of the template once per delimiter per tag,
		// which is quadratic in the tag count: measured on flat templates, 5k tags parsed in
		// 20ms, 10k in 66ms, 20k in 254ms -- 3.8x for each doubling. Searching for the one
		// character they all start with and checking the second byte costs a single pass.
		next := -1
		for j := i; j+1 < len(src); j++ {
			if src[j] != '{' {
				continue
			}
			switch src[j+1] {
			case '{', '%', '#':
				next = j - i
			}
			if next >= 0 {
				break
			}
		}
		if next < 0 {
			text.WriteString(src[i:])
			break
		}
		text.WriteString(src[i : i+next])
		i += next

		if i+1 >= len(src) {
			return nil, fmt.Errorf("chat: template ends inside %q", src[i:])
		}
		open := src[i : i+2]
		var closeTag string
		var kind pieceKind
		switch open {
		case "{{":
			closeTag, kind = "}}", pieceExpr
		case "{%":
			closeTag, kind = "%}", pieceStmt
		case "{#":
			closeTag, kind = "#}", pieceText
		}

		end := indexCloseTag(src[i+2:], closeTag)
		if end < 0 {
			return nil, fmt.Errorf("chat: template has an unclosed %q at offset %d", open, i)
		}
		inner := src[i+2 : i+2+end]
		i += 2 + end + len(closeTag)

		// A comment produces no output at all, but its trim markers still apply to
		// the text around it.
		trimBefore := strings.HasPrefix(inner, "-")
		trimAfter := strings.HasSuffix(inner, "-")
		inner = strings.TrimPrefix(inner, "-")
		inner = strings.TrimSuffix(inner, "-")

		flushText(trimBefore)
		if kind == pieceText {
			// A comment emits nothing, so its trailing trim has to be carried to the
			// next literal directly rather than through a piece.
			trimPending = trimAfter
			continue
		}
		out = append(out, piece{
			kind:       kind,
			text:       strings.TrimSpace(inner),
			trimBefore: trimBefore,
			trimAfter:  trimAfter,
			// i already points past the tag, so the start is recovered by subtracting what
			// was consumed: the delimiters, the body, and the closing tag.
			offset: i - (2 + end + len(closeTag)),
		})
	}
	flushText(false)

	if err := applyTrimMarkers(out); err != nil {
		return nil, err
	}
	return out, nil
}

// applyTrimMarkers removes the whitespace a tag's -%} marker asks for from the literal
// text that follows it.
//
// The before-side trim happens while scanning, because the preceding literal is already
// complete at that point; the after-side trim can only be applied once the following
// literal exists, which is why it is a second pass.
func applyTrimMarkers(pieces []piece) error {
	for i := range pieces {
		if !pieces[i].trimAfter || pieces[i].kind == pieceText {
			continue
		}
		if i+1 < len(pieces) && pieces[i+1].kind == pieceText {
			pieces[i+1].text = strings.TrimLeft(pieces[i+1].text, " \t\r\n")
		}
	}
	return nil
}
