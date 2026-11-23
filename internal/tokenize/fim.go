package tokenize

import "slices"

// FIM is the fill-in-the-middle marker set a vocabulary declares.
//
// A code-tuned checkpoint is trained on a prompt assembled as prefix, then suffix, then a
// marker that opens the part to be filled in, which is how a completion is produced for text
// inserted into the middle of a document. The markers are ordinary vocabulary entries with
// names a family chose, so the only way to find them is to look for the names.
type FIM struct {
	// Prefix, Suffix and Middle are the three marker ids that frame the prompt.
	Prefix, Suffix, Middle int32
	// Names are the same markers as their vocabulary pieces, in the order Prefix, Suffix,
	// Middle.
	Names [3]string
	// End and EndNames are the markers that close an insertion, as ids and as pieces.
	//
	// They are separate from the three above because they are not part of the prompt: the
	// model emits one of them to say the insertion is finished, and generation has to stop
	// there or the reply continues into text that belongs after the insertion point. Measured
	// on Qwen2.5-Coder-0.5B without them: the correct expression was generated first and the
	// response then carried on writing the caller's own following lines.
	End      []int32
	EndNames []string
}

// fimMarkerSet is one family's spelling.
type fimMarkerSet struct {
	prefix, suffix, middle string
	// end lists the pieces that may close an insertion, in the order they are worth trying.
	// A family that declares none relies on its end-of-text token, which the engine stops on
	// already; a name absent from a vocabulary is simply skipped.
	end []string
}

// fimMarkerSets lists the marker names model families use, most specific first.
//
// Only names are matched, never positions: two families can put their markers at different
// ends of a vocabulary, and the Qwen and StarCoder spellings differ only by the pipes around
// the name. A set counts as declared only if the vocabulary holds all three framing markers,
// because a Prefix without a Middle describes no prompt that could be built.
var fimMarkerSets = []fimMarkerSet{
	{
		// Qwen2.5-Coder and CodeGemma.
		prefix: "<|fim_prefix|>", suffix: "<|fim_suffix|>", middle: "<|fim_middle|>",
		end: []string{"<|fim_pad|>", "<|endoftext|>"},
	},
	{
		// StarCoder and StarCoder2.
		prefix: "<fim_prefix>", suffix: "<fim_suffix>", middle: "<fim_middle>",
		end: []string{"<fim_pad>", "<|endoftext|>"},
	},
	{
		// CodeLlama.
		prefix: "<PRE>", suffix: "<SUF>", middle: "<MID>",
		end: []string{"<EOT>"},
	},
	{
		// SentencePiece spellings, where the marker carries the word-boundary space.
		prefix: "\u2581<fim_prefix>", suffix: "\u2581<fim_suffix>", middle: "\u2581<fim_middle>",
		end: []string{"<|fim_pad|>", "<|endoftext|>"},
	},
}

// fimMarkerSetNames returns the family spellings, for a caller that has to name them.
func fimMarkerSetNames() [][3]string {
	out := make([][3]string, len(fimMarkerSets))
	for i, set := range fimMarkerSets {
		out[i] = [3]string{set.prefix, set.suffix, set.middle}
	}
	return out
}

// FIMMarkerNames lists the framing marker sets FIMTokens looks for, so a caller refusing a
// request can say what the served checkpoint would have needed to declare.
func FIMMarkerNames() [][3]string {
	return fimMarkerSetNames()
}

// FIMTokens reports the fill-in-the-middle markers this vocabulary declares, and whether it
// declares a usable set.
//
// A vocabulary with none is not a failure: most checkpoints are not trained for it, and the
// caller's job is to refuse the request honestly rather than to assemble a prompt from markers
// the model has never seen -- which produces plausible text in the right shape and the wrong
// content.
//
// Declaring the markers is necessary but not sufficient. A vocabulary is inherited from the
// tokenizer a model was converted with, so an instruct checkpoint can carry a code model's
// markers without ever having been trained to use them. Measured on Qwen2.5-0.5B-Instruct,
// which declares all three: the assembled prompt is well formed and the reply ignores it,
// repeating the suffix instead of filling it. Nothing in a GGUF file separates the two cases,
// so this reports what the file says and the caller decides whether to trust it.
//
// The scan is one pass over the vocabulary per call. It is not cached because it is consulted
// once per request that asks for an insertion, and a cache would need invalidating if a
// tokenizer were ever reloaded in place.
func (t *Tokenizer) FIMTokens() (FIM, bool) {
	// One lookup table of every candidate name, so the pass over the vocabulary is one map hit
	// per token rather than a comparison per name.
	wanted := make(map[string]struct{}, len(fimMarkerSets)*6)
	for _, set := range fimMarkerSets {
		for _, name := range append([]string{set.prefix, set.suffix, set.middle}, set.end...) {
			wanted[name] = struct{}{}
		}
	}
	found := make(map[string]int32, len(wanted))
	for i, piece := range t.tokens {
		if _, want := wanted[piece]; !want {
			continue
		}
		// The first occurrence wins. A vocabulary that lists a piece twice is damaged, and
		// either id encodes the same marker; taking the first keeps the choice deterministic
		// rather than whatever the map happened to see last.
		if _, seen := found[piece]; !seen {
			found[piece] = int32(i)
		}
	}

	for _, set := range fimMarkerSets {
		p, okP := found[set.prefix]
		s, okS := found[set.suffix]
		m, okM := found[set.middle]
		if !okP || !okS || !okM {
			continue
		}
		fim := FIM{
			Prefix: p, Suffix: s, Middle: m,
			Names: [3]string{set.prefix, set.suffix, set.middle},
		}
		for _, name := range set.end {
			id, ok := found[name]
			if !ok {
				continue
			}
			fim.End = append(fim.End, id)
			fim.EndNames = append(fim.EndNames, name)
		}
		return fim, true
	}
	return FIM{}, false
}

// StopIDs returns every marker generation should stop on, framing and closing together.
//
// The ids are what the engine stops on, and they have to be ids: the markers are control
// tokens, which encode to themselves only when matched whole, so a caller that passed their
// text through the ordinary encoder would get the characters of the marker instead of the
// token and stop on nothing.
func (f FIM) StopIDs() []int32 {
	out := make([]int32, 0, 3+len(f.End))
	out = append(out, f.Prefix, f.Suffix, f.Middle)
	return append(out, f.End...)
}

// StopStrings returns the same markers as text, which is what a caller adds to whatever the
// request already stops on.
//
// They are returned as text as well as ids because the response is cut on decoded text: a
// marker that reached the text at all has to be removed there, whether it arrived as its own
// token or inside a byte run.
func (f FIM) StopStrings() []string {
	out := slices.Clone(f.Names[:])
	for _, name := range f.EndNames {
		if !slices.Contains(out, name) {
			out = append(out, name)
		}
	}
	slices.Sort(out)
	return out
}
