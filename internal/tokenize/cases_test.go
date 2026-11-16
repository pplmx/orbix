package tokenize

// spmCases are tokenizations produced by llama.cpp's own SPM tokenizer
// (llm_tokenizer_spm_session in src/llama-vocab.cpp) for the vocabulary in
// fixture_test.go. That is the implementation a GGUF model is actually
// served by, so matching it -- not textbook SentencePiece unigram Viterbi --
// is what makes Orbix produce the tokens the model was trained with.
var spmCases = []struct {
	text string
	ids  []int32
}{
	{
		text: "Once upon a time",
		ids:  []int32{3, 35, 4, 18, 5, 3, 27, 32, 11, 4, 10, 17, 7, 22},
	},
	{
		text: " there was a little girl",
		ids:  []int32{3, 13, 28, 16, 8, 10, 3, 6, 7, 9, 9, 6, 5, 42, 34, 6},
	},
	{
		text: "The cat sat on the mat.",
		ids:  []int32{36, 20, 3, 18, 31, 45, 31, 3, 11, 4, 13, 3, 49, 31, 12},
	},
	{
		text: "Tom went to the park with his dog.",
		ids:  []int32{36, 11, 49, 26, 5, 4, 9, 17, 11, 13, 29, 57, 19, 26, 7, 44, 3, 30, 7, 8, 3, 14, 11, 33, 12},
	},
	{
		text: "  double  spaces  ",
		ids:  []int32{3, 3, 3, 14, 11, 27, 59, 6, 5, 3, 45, 32, 21, 18, 5, 8, 3, 3},
	},
	{
		text: "Hello, world!",
		ids:  []int32{3, 50, 5, 6, 6, 11, 0, 26, 11, 47, 6, 14, 0},
	},
	{
		text: "a",
		ids:  []int32{10},
	},
	{
		text: "unknown character: @#$%",
		ids:  []int32{3, 41, 19, 4, 11, 48, 4, 3, 18, 30, 57, 21, 18, 9, 5, 47, 0, 3, 0, 0, 0, 0},
	},
	{
		text: "Lily liked to play in the garden every day.",
		ids:  []int32{3, 52, 7, 6, 15, 3, 6, 7, 19, 5, 14, 17, 11, 29, 6, 40, 3, 7, 4, 13, 42, 57, 14, 5, 4, 3, 5, 54, 5, 47, 15, 3, 14, 40, 12},
	},
}
