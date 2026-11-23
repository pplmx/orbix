package tokenize

import (
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
)

// TestFIMTokensFindsTheDeclaredSet covers detection for a vocabulary that has the markers.
//
// The names are matched, never positions, so the test puts them at unrelated indices: an
// implementation that looked for a family's usual numbering would pass on a checkpoint that
// happened to match and fail on every conversion that ordered the vocabulary differently.
func TestFIMTokensFindsTheDeclaredSet(t *testing.T) {
	vocab := []string{
		"<unk>", "<s>", "</s>",
		"hello", "<|fim_middle|>", "world", "<|fim_prefix|>", "x", "<|fim_suffix|>",
	}
	tok := bpeFixture(t, gguftest.Model{Vocab: vocab})

	fim, ok := tok.FIMTokens()
	if !ok {
		t.Fatal("a vocabulary declaring all three markers reported none")
	}
	want := map[string]int32{}
	for i, piece := range vocab {
		want[piece] = int32(i)
	}
	if fim.Prefix != want["<|fim_prefix|>"] {
		t.Errorf("Prefix = %d, want %d", fim.Prefix, want["<|fim_prefix|>"])
	}
	if fim.Suffix != want["<|fim_suffix|>"] {
		t.Errorf("Suffix = %d, want %d", fim.Suffix, want["<|fim_suffix|>"])
	}
	if fim.Middle != want["<|fim_middle|>"] {
		t.Errorf("Middle = %d, want %d", fim.Middle, want["<|fim_middle|>"])
	}
	// The names come back as pieces, because generation stops on them as text.
	if got, want3 := fim.Names, [3]string{"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"}; got != want3 {
		t.Errorf("Names = %v, want %v", got, want3)
	}
}

// TestFIMTokensRequiresAllThree pins the completeness rule.
//
// A Prefix without a Middle describes no prompt that could be built, and answering with a
// half-assembled one would produce text that looks like a completion and is not one. Each
// partial set here is one marker short of a real family's.
func TestFIMTokensRequiresAllThree(t *testing.T) {
	for _, missing := range []string{"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"} {
		vocab := []string{"<unk>", "<s>", "</s>", "a"}
		for _, marker := range []string{"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"} {
			if marker != missing {
				vocab = append(vocab, marker)
			}
		}
		t.Run("without "+missing, func(t *testing.T) {
			tok := bpeFixture(t, gguftest.Model{Vocab: vocab})
			if _, ok := tok.FIMTokens(); ok {
				t.Errorf("reported a usable set while %s is absent", missing)
			}
		})
	}
}

// TestFIMTokensRecognisesEachFamily covers the marker spellings, because a checkpoint uses only
// its own and detection that knew one of them would silently refuse every other code model.
func TestFIMTokensRecognisesEachFamily(t *testing.T) {
	cases := []struct {
		name  string
		vocab []string
	}{
		{"qwen and codegemma", []string{
			"<unk>", "<s>", "</s>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
		}},
		{"starcoder", []string{
			"<unk>", "<s>", "</s>", "<fim_prefix>", "<fim_suffix>", "<fim_middle>",
		}},
		{"codellama", []string{
			"<unk>", "<s>", "</s>", "<PRE>", "<SUF>", "<MID>",
		}},
		{"sentencepiece", []string{
			"<unk>", "<s>", "</s>", "\u2581<fim_prefix>", "\u2581<fim_suffix>", "\u2581<fim_middle>",
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tok := bpeFixture(t, gguftest.Model{Vocab: c.vocab})
			fim, ok := tok.FIMTokens()
			if !ok {
				t.Fatalf("%s markers were not recognised", c.name)
			}
			// All three ids must be distinct and inside the vocabulary.
			ids := []int32{fim.Prefix, fim.Suffix, fim.Middle}
			for i := range ids {
				for j := i + 1; j < len(ids); j++ {
					if ids[i] == ids[j] {
						t.Errorf("two markers share the id %d", ids[i])
					}
				}
			}
		})
	}
}

// TestFIMTokensOnAVocabularyWithoutMarkers is the case every Llama checkpoint hits.
//
// It has to be reported as absent rather than guessed at: assembling a prompt from markers a
// model never saw produces fluent text in the shape of an answer, which is worse than a refusal
// because nothing about it looks wrong.
func TestFIMTokensOnAVocabularyWithoutMarkers(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{})
	if fim, ok := tok.FIMTokens(); ok {
		t.Errorf("an ordinary vocabulary reported markers: %+v", fim)
	}
}

// TestFIMStopSetsIncludeTheClosingMarkers checks both halves of the stop list, which is how a
// generated insertion is ended.
//
// The framing markers alone are not enough. A model closes its insertion by emitting a
// dedicated end marker -- measured on Qwen2.5-Coder, `<|fim_pad|>` or `<|endoftext|>` -- and
// without those in the stop list the reply carries on into text that belongs after the
// insertion point. Measured: the correct expression was generated first and the response then
// kept writing the caller's own following lines, because nothing told it to stop.
//
// Both forms are returned: ids for the engine, which stops on a token, and text for the cut,
// which happens after decoding. The ids have to be ids because the markers are control tokens
// that the ordinary encoder would spell as characters rather than match whole.
func TestFIMStopSetsIncludeTheClosingMarkers(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{Vocab: []string{
		"<unk>", "<s>", "</s>",
		"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|fim_pad|>", "<|endoftext|>",
	}})
	fim, ok := tok.FIMTokens()
	if !ok {
		t.Fatal("markers not found")
	}

	if len(fim.End) != 2 {
		t.Errorf("got %d closing markers, want 2 (the pad and end-of-text pieces)", len(fim.End))
	}
	// The framing three and every closing marker the vocabulary declares, with no duplicates.
	ids := fim.StopIDs()
	if len(ids) != 5 {
		t.Errorf("StopIDs returned %d ids, want 5: %v", len(ids), ids)
	}
	seen := make(map[int32]bool, len(ids))
	for _, id := range ids {
		if seen[id] {
			t.Errorf("id %d appears twice in the stop set", id)
		}
		seen[id] = true
	}

	strs := fim.StopStrings()
	want := []string{"<|endoftext|>", "<|fim_middle|>", "<|fim_pad|>", "<|fim_prefix|>", "<|fim_suffix|>"}
	if len(strs) != len(want) {
		t.Fatalf("StopStrings returned %v, want %v", strs, want)
	}
	for i := range want {
		if strs[i] != want[i] {
			t.Errorf("StopStrings[%d] = %q, want %q", i, strs[i], want[i])
		}
	}
}

// TestFIMWithoutClosingMarkersStillWorks covers a family that declares no end marker.
//
// CodeLlama's `<EOT>` is not in every conversion, and a vocabulary holding only the framing
// three has to give a usable set rather than none: the engine still stops on the model's
// end-of-text token, so the request is answerable even without a dedicated closing marker.
func TestFIMWithoutClosingMarkersStillWorks(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{Vocab: []string{
		"<unk>", "<s>", "</s>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
	}})
	fim, ok := tok.FIMTokens()
	if !ok {
		t.Fatal("framing markers without a closing marker must still be usable")
	}
	if len(fim.End) != 0 {
		t.Errorf("got closing markers %v from a vocabulary holding none", fim.EndNames)
	}
	if got := len(fim.StopIDs()); got != 3 {
		t.Errorf("StopIDs returned %d ids, want the three framing markers", got)
	}
	if got := len(fim.StopStrings()); got != 3 {
		t.Errorf("StopStrings returned %d strings, want the three framing markers", got)
	}
}
