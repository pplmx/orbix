// Code generated from llama.cpp's own expressions; DO NOT EDIT.
//
// Each string here is one the Go implementation got wrong at some point, so the table
// doubles as a regression list. Expectations come from applying llama.cpp's expressions
// as sequential passes, not from reading the Go code, so a failure means the rule drifted
// from the reference.

package tokenize

// preReferenceCases lists, per family, the chunks that family's own llama.cpp expressions
// produce for inputs that have been wrong here before.
var preReferenceCases = []struct {
	name  string
	pre   preType
	cases map[string][]string
}{
	{
		name: "deepseek-coder",
		pre:  preDeepseekCoder,
		cases: map[string][]string{
			"\r\r":     {"\r", "\r"},
			")क\r\r":   {")", "क", "\r", "\r"},
			"9\tก":     {"9", "\t", "ก"},
			"a\r\nb":   {"a", "\r", "\n", "b"},
			"b - ":     {"b", " ", "-", " "},
			"́`กA _%[": {"́`", "ก", "A", " _%["},
		},
	},
	{
		name: "deepseek-llm",
		pre:  preDeepseekLLM,
		cases: map[string][]string{
			";]&.%́  !": {";]&.%", "́", " ", " ", "!"},
			"a b":       {"a", " b"},
			"́  !":      {"́", " ", " ", "!"},
			"‌中 ":       {"‌中", " "},
			"😀\t":       {"😀", "\t"},
		},
	},
	{
		name: "falcon",
		pre:  preFalcon,
		cases: map[string][]string{
			"  123":    {" ", " ", "123"},
			"1234":     {"123", "4"},
			"1234567²": {"123", "456", "7²"},
			"99²":      {"99²"},
			"`  Ⅳ0Ⅳ[":  {"`", " ", " Ⅳ0Ⅳ", "["},
			"ⅣⅣⅣ":      {"ⅣⅣⅣ"},
		},
	},
	{
		name: "flint",
		pre:  preFlint,
		cases: map[string][]string{
			"'re23's's":      {"'re", "23", "'s's"},
			"'tcamel":        {"'tcamel"},
			".23'tcamelCase": {".", "23", "'tcamel", "Case"},
			"bก":             {"bก"},
			"camelCase":      {"camel", "Case"},
			"don't":          {"don't"},
			"文A":             {"文", "A"},
		},
	},
	{
		name: "jais2",
		pre:  preJais2,
		cases: map[string][]string{
			"\t.\t中":   {"\t", ".", "\t中"},
			"  .中1":    {" ", " .", "中", "1"},
			"  's.":    {" ", " ", "'s", "."},
			"'s中  ́  ": {"'s", "中", " ", " ́", "  "},
			".\t23A\n": {".", "\t", "23", "A", "\n"},
			"a ":       {"a", " "},
			"aAá.":    {"aAa", "́."},
			" ́́.":     {" ", "́́."},
			"́.'s²":    {"́.'", "s", "²"},
			" 2323":    {" ", "232", "3"},
			"中\t   ":   {"中", "\t   "},
			"中\n":      {"中", "\n"},
		},
	},
	{
		name: "laguna",
		pre:  preLaguna,
		cases: map[string][]string{
			"\tA'sa":   {"\tA", "'s", "a"},
			" ²² ":     {" ", "²", "²", " "},
			"'s ":      {"'s", " "},
			".'s中a ":   {".'", "s中a", " "},
			"1́1  ":    {"1", "́", "1", "  "},
			"1‌":       {"1", "‌"},
			"23. \t's": {"2", "3", ".", " ", "\t", "'s"},
			"23A":      {"2", "3", "A"},
			"23AA²中":   {"2", "3", "AA", "²", "中"},
			"A中中":      {"A中中"},
			"a² A":     {"a", "²", " A"},
			" 's.":     {" ", "'s", "."},
		},
	},
	{
		name: "minicpm5",
		pre:  preMiniCPM5,
		cases: map[string][]string{
			"\t1's23": {"\t", "1", "'s", "23"},
			"\ta A":   {"\ta", " A"},
			". a":     {".", " a"},
			"2323  ":  {"232", "3", "  "},
			"AA\n":    {"AA", "\n"},
			"Aa‌23  ": {"Aa", "‌", "23", "  "},
			"A 's":    {"A", " ", "'s"},
			"a‌ ":     {"a", "‌", " "},
			"́ 's  ":  {"́", " ", "'s", "  "},
			"  ":      {"  "},
			"‌Á\t":   {"‌A", "́", "\t"},
			"中 ":      {"中", " "},
		},
	},
	{
		name: "soprano",
		pre:  preSoprano,
		cases: map[string][]string{
			"\t\n‌‌":  {"\t\n", "‌‌"},
			" \t ²a":  {" \t ", "²", "a"},
			"  A":     {"  ", "A"},
			" ́‌":     {" ", "́‌"},
			" ‌":      {" ", "‌"},
			".  ":     {".", "  "},
			". 's":    {".", " ", "'", "s"},
			"23\n.'s": {"2", "3", "\n", ".'", "s"},
			"a.‌²":    {"a", ".‌", "²"},
			"a‌":      {"a", "‌"},
			"²  ́23":  {"²", "  ", "́", "2", "3"},
			"²́ \t中":  {"²", "́", " \t", "中"},
		},
	},
	{
		name: "spark2_5",
		pre:  preSpark25,
		cases: map[string][]string{
			"\n\n":   {"\n", "\n"},
			"\r\n":   {"\r", "\n"},
			"'sa":    {"'sa"},
			",́":     {",", "́"},
			"a\r\nb": {"a", "\r", "\n", "b"},
			"don't":  {"don", "'t"},
		},
	},
	{
		name: "superbpe",
		pre:  preSuperBPE,
		cases: map[string][]string{
			"111aa1": {"111", "aa", "1"},
			"a 111":  {"a ", "111"},
		},
	},
	{
		name: "tekken",
		pre:  preTekken,
		cases: map[string][]string{
			"ABc":        {"ABc"},
			"HTTPServer": {"HTTPServer"},
			"aB":         {"a", "B"},
			"bก":         {"bก"},
			"helloWorld": {"hello", "World"},
			"中中a1":       {"中中a", "1"},
			"文A":         {"文", "A"},
		},
	},
	{
		name: "tiny_aya",
		pre:  preTinyAya,
		cases: map[string][]string{
			"0‍":      {"0", "‍"},
			"0Ⅳ":      {"0Ⅳ"},
			"1234":    {"1", "234"},
			"1234567": {"1", "234", "567"},
			"1²":      {"1", "²"},
			"a1234b":  {"a", "123", "4", "b"},
			"Ⅳ١Ⅳ":     {"Ⅳ١Ⅳ"},
		},
	},
	{
		name: "ufakzeka",
		pre:  preUfakzeka,
		cases: map[string][]string{
			"\n.1":    {"\n", ".", "1"},
			"1  Á ":  {"1", " ", " A", "́", " "},
			"A23A":    {"A", "2", "3", "A"},
			"A Á‌":   {"A", " A", "́‌"},
			"A‌中aA":   {"A", "‌中aA"},
			"²'ś.\n": {"²", "'s", "́.\n"},
			"²A.\t.":  {"²", "A", ".", "\t", "."},
			" 23 a":   {" ", "2", "3", " a"},
			"   a":    {"  ", " a"},
			"‌A":      {"‌A"},
			"‌中":      {"‌中"},
			"中1́":     {"中", "1", "́"},
		},
	},
	{
		name: "viking",
		pre:  preViking,
		cases: map[string][]string{
			";=*{中 ":  {";=*{中", " "},
			"abc def": {"abc", " def"},
			"a b":     {"a", " ", "b"},
			"a b":     {"a", " ", "b"},
		},
	},
	{
		name: "youtu",
		pre:  preYoutu,
		cases: map[string][]string{
			"\t²A":    {"\t", "²", "A"},
			"\t  a中":  {"\t ", " a", "中"},
			"  1\t  ": {" ", " ", "1", "\t  "},
			"'s1A\t":  {"'s", "1", "A", "\t"},
			".23 23":  {".", "2", "3", " ", "2", "3"},
			"23A":     {"2", "3", "A"},
			"23²":     {"2", "3", "²"},
			"23‌":     {"2", "3", "‌"},
			"23中23a‌": {"2", "3", "中", "2", "3", "a", "‌"},
			"a中23":    {"a", "中", "2", "3"},
			" ².":     {" ", "²", "."},
			"²́ 1":    {"²", "́", " ", "1"},
		},
	},
	{
		name: "zonos3",
		pre:  preZonos3,
		cases: map[string][]string{
			"\r 😀9!中":  {"\r 😀", "9", "!中"},
			";\r\n‌9*": {";\r\n‌", "9", "*"},
			";;":       {";;"},
			"‌":        {"‌"},
		},
	},
}
