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
package tokenize
