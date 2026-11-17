// Package tokenize converts text to token IDs and back using the vocabulary stored in a
// GGUF file.
//
// Two tokenizer families appear in GGUF models, and this package implements both.
// SentencePiece models (Llama, Mistral, Gemma) store a vocabulary plus a per-token score
// and segment text by repeatedly merging the highest-scoring adjacent pair. Byte-level BPE
// models (GPT-2 lineage, Qwen, Llama 3) store an explicit merge list instead, and that
// list's order is the tokenization.
//
// The SentencePiece path follows llama.cpp's llm_tokenizer_spm rather than textbook
// SentencePiece, which uses Viterbi decoding over the unigram language model. The two
// agree on ordinary text but can differ where several segmentations score similarly;
// matching llama.cpp is what makes a GGUF model produce the tokens it was trained with.
//
// # Decisions worth knowing about
//
// A merge list is applied in rank order, not by any score. Lower rank merges first, and a
// pair absent from the list never merges however its pieces score. This is not a detail:
// the official Qwen2 checkpoint carries 151387 rules, and scoring the pieces instead -- the
// unigram rule, a different algorithm -- produces valid token ids and fluent text on
// ordinary prose while tokenizing roughly a third of unusual inputs differently. The
// difference is visible where it matters: on "antidisestablishmentarianism is a word that
// means", the score path generated text about an "anti-disaster movement" -- the word was
// split wrong, so the model answered a different question -- while applying the list gives
// "to establish a new religion".
//
// Pre-tokenization is a separate question from the merge algorithm. A byte-level BPE model
// was trained on chunks, not on raw text: the trainer cut the corpus first and a merge never
// spans a boundary, so reproducing the vocabulary's expectations means applying the same cut
// before merging. tokenizer.ggml.pre names the cut, and llama.cpp recognises about a hundred
// names for it.
//
// Thirty-two families are implemented here, covering 98 of those names, and the list is
// per-family work rather than aliasing: a name is either flagged implemented or the file it
// describes is split as one chunk and says so. The first few were chosen because they cover
// the models this project has been run against; the rest because each added a dimension the
// others did not have. Most families share a template -- contractions, a letter run, a digit
// run, a punctuation run, whitespace -- and differ in choices, so the choices are data in
// preRule rather than a function each.
//
// deepseek-v3 brought a letter class that includes combining marks, a punctuation
// alternative naming categories instead of their complement, and a digit-group and CJK
// expression ahead of the main rule. tekken brought structure *inside* the letter run: a run
// is cut where a lower-case letter is followed by an upper-case one, so "camelCase" is two
// chunks while "HTTPServer" -- a capital run with its trailing word -- stays one. Both added
// fields to preRule and neither needed a second code path, which is the test the
// parameterisation was built to pass.
//
// Some families need their own code path, and the reason is the same for most of them: their
// expressions are applied in sequence over the pieces the previous one left, so they are
// passes rather than alternatives. deepseek-coder, deepseek-llm, falcon and chameleon share
// that shape; chameleon is the sharpest case, because its punctuation pass cuts single
// characters ahead of the main rule and so an apostrophe never forms a contraction. Reading
// its expression as one alternatives list would put boundaries in the wrong place on
// ordinary text -- "don't" would keep its "'t" -- and nothing would report it.
//
// Everything else splits as one chunk. That is the correct answer for a file naming nothing
// and for one naming "default", because llama.cpp applies no expression in those cases
// either. For a family that exists and is not implemented it is the wrong answer, and
// silently so: the text still tokenizes to valid ids, just to ids the model was not trained
// to expect, so the effect is degraded output rather than an error. PreImplemented and
// PreKnown tell those cases apart, and "orbix model inspect" prints the answer, because a
// reader judging a model's quality needs to know the prompt was cut wrong before they blame
// the model.
//
// The two families represent a space differently, and the difference is silent when wrong.
// SentencePiece folds a space into a U+2581 marker on the following piece, so " a" is one
// symbol; byte-level BPE escapes it as its own symbol and lets a merge decide. Using the
// wrong convention does not fail: every word still encodes to valid ids and every space
// disappears, which is why the convention is read from tokenizer.ggml.model rather than
// assumed.
//
// Control tokens are matched whole, before any other segmentation. A chat template is not
// text: "<|im_start|>" is one token the model was trained to see, and encoding it as the
// six ordinary pieces it happens to be spelled from gives the model a prompt it has never
// seen -- it answers as though the marker were prose.
//
// Decoding a stream and decoding a whole sequence are not the same function, and the
// difference is real rather than an oversight. Decode joins every pending byte before
// deciding how to render a run; a stream must decide when the run has ended, so a byte
// sequence that never forms a character can render differently. Text a model actually
// generates does not produce such runs.
//
// Decoding keeps the two vocabulary families apart for the same reason encoding does. A
// SentencePiece piece is plain text plus the space marker; a byte-level piece is a rune per
// byte. Rendering a SentencePiece piece through the byte table is not a no-op: once the
// table covered the whole range, a Spanish "¡" in a SentencePiece vocabulary decoded to a
// lone byte, because the piece's own character looked like an escape.
//
// Decode is the exact inverse of Encode, which is what makes a round trip a usable check.
// That means it keeps a leading space the caller wrote. Rendering a completion is a
// different job: a byte-level vocabulary spells a word two ways ("the" and "\u0120the"), and a
// generated continuation starts a word with the spaced form, so the space is not something
// the user asked to see. DecodeCompletion is the wrapper for that, and the stream decoder
// applies the same rule inline because a stream has no whole sequence left to post-process.
// llama.cpp splits the same way, with an lstrip argument to token_to_piece rather than logic
// inside its detokenize.
package tokenize
