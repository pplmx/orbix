// Package tokenize converts text to token IDs and back using the vocabulary
// stored in a GGUF file.
//
// Two tokenizer families appear in GGUF models. SentencePiece models (Llama,
// Mistral, Gemma) store a vocabulary plus a per-token score, and segment text
// by repeatedly merging the highest-scoring adjacent pair. BPE models (GPT-2
// lineage, Qwen) store an explicit merge list instead. This package implements
// the SentencePiece family, which is what the models Orbix targets use.
//
// The merge-based algorithm here follows llama.cpp's llm_tokenizer_spm rather
// than textbook SentencePiece, which uses Viterbi decoding over the unigram
// language model. The two agree on ordinary text but can differ on inputs
// where several segmentations score similarly; matching llama.cpp is what
// makes a GGUF model produce the same tokens it was trained and evaluated
// with.
package tokenize
