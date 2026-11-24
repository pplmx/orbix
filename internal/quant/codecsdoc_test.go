package quant

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestModelsDocListsEveryDecodableFormat keeps the quantization table and codecs in step.
//
// docs/models.md tells a reader which quantizations their checkpoint may use. codecs is the same
// list in code: Decode looks the tensor's type up there and refuses anything absent with
// "cannot decode". A format the document omits understates what the project can load; one it names
// that codecs does not have promises a file that will be refused after the whole download.
func TestModelsDocListsEveryDecodableFormat(t *testing.T) {
	doc, err := os.ReadFile(filepath.Join("..", "..", "docs", "models.md"))
	if err != nil {
		t.Fatalf("reading docs/models.md: %v", err)
	}
	documented := documentedFormats(t, string(doc))
	if len(documented) < 10 {
		t.Fatalf("read %d formats from the document, too few to be the real table -- the parser "+
			"probably stopped matching", len(documented))
	}

	// Names as the document spells them, from gguf's own String method so a renamed type would
	// show up here rather than in a hand-maintained map.
	inCode := make(map[string]bool, len(codecs))
	for typ := range codecs {
		inCode[typ.String()] = true
	}

	for name := range inCode {
		if !documented[name] {
			t.Errorf("codecs decodes %s, which docs/models.md does not list", name)
		}
	}
	for name := range documented {
		if !inCode[name] {
			t.Errorf("docs/models.md lists %s, which codecs cannot decode -- a file using it is "+
				"refused at load", name)
		}
	}
}

// documentedFormats reads the quantization table's first column.
//
// A cell may hold several names separated by commas, and the K-quants row writes a range --
// "`K-quants` (`Q2_K`-`Q6_K`)" -- because five entries differing only in the middle digit read
// better that way. The range is expanded here rather than in the document: the check is about the
// set of formats, and a document that spelled all five out would be no more correct.
func documentedFormats(t *testing.T, text string) map[string]bool {
	t.Helper()
	out := make(map[string]bool)
	inTable := false
	for line := range strings.SplitSeq(text, "\n") {
		if !strings.HasPrefix(line, "|") {
			if inTable {
				break
			}
			continue
		}
		cells := strings.Split(strings.Trim(line, "|"), "|")
		if len(cells) < 2 {
			continue
		}
		first := strings.TrimSpace(cells[0])
		if first == "Quantization" {
			inTable = true
			continue
		}
		if !inTable || strings.HasPrefix(first, "--") {
			continue
		}
		for _, m := range tableFormat.FindAllStringSubmatch(first, -1) {
			out[m[1]] = true
		}
		// A range: two names whose leading integer differs.
		if m := tableRange.FindStringSubmatch(first); m != nil && m[2] == m[4] {
			// Both ends must share a suffix for the range to mean anything; Q2_K-Q6_K does,
			// and a cell pairing unlike names is a typo the format check below will catch.
			for i := m[1][0]; i <= m[3][0]; i++ {
				out["Q"+string(i)+"_"+m[2]] = true
			}
		}
	}
	return out
}

// tableFormat matches one format name in a table cell.
var tableFormat = regexp.MustCompile("`([A-Z][A-Z0-9_]+)`")

// tableRange matches "`Q2_K`-`Q6_K`", capturing both digits and both suffixes so the caller can
// check the suffixes match. Go's regexp is RE2 and has no back references, so `\2` is not available
// and the comparison happens in code.
var tableRange = regexp.MustCompile("`Q(\\d)_([A-Z]+)`-`Q(\\d)_([A-Z]+)`")
