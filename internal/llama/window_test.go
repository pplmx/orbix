package llama

import (
	"math"
	"strings"
	"testing"
)

// trimSpans is the whole sliding-window mechanism: if it keeps the wrong rows, every
// Mistral decode quietly attends over the wrong positions. It has to handle a single
// contiguous span, several paged spans, and a cut inside one of them.

// makeSpans builds spans whose row i holds value base+i, one value per row, so a test can
// tell which positions survived a trim.
func makeSpans(groups ...int) ([]span, []float32) {
	var data []float32
	var spans []span
	next := 0
	for _, rows := range groups {
		start := len(data)
		for i := range rows {
			data = append(data, float32(next+i))
		}
		spans = append(spans, span{
			data:   data[start:],
			stride: 1,
			rows:   rows,
			width:  1,
		})
		next += rows
	}
	return spans, data
}

// spanValues returns the surviving rows' values in order.
func spanValues(spans []span) []float32 {
	var out []float32
	for _, s := range spans {
		for i := range s.rows {
			out = append(out, s.row(i)[0])
		}
	}
	return out
}

func TestTrimSpansKeepsLastRows(t *testing.T) {
	tests := []struct {
		name   string
		groups []int
		keep   int
		want   []float32
	}{
		{"whole span, no trim", []int{10}, 10, seq(0, 10)},
		{"whole span, trim to 3", []int{10}, 3, []float32{7, 8, 9}},
		{"keep one", []int{10}, 1, []float32{9}},
		{"keep all plus slack", []int{10}, 20, seq(0, 10)},
		{"two spans, cut in first", []int{4, 6}, 8, seq(2, 8)},
		{"two spans, cut exactly at boundary", []int{4, 6}, 6, seq(4, 6)},
		{"three spans, cut in middle", []int{4, 4, 4}, 6, seq(6, 6)},
		{"keep zero", []int{10}, 0, nil},
		{"empty", nil, 3, nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			spans, _ := makeSpans(tc.groups...)
			got := spanValues(trimSpans(spans, tc.keep))
			if len(got) != len(tc.want) {
				t.Fatalf("kept %d rows %v, want %d rows %v", len(got), got, len(tc.want), tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("row %d = %v, want %v (kept %v)", i, got[i], tc.want[i], got)
				}
			}
		})
	}
}

// seq returns n consecutive floats starting at base.
func seq(base, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(base + i)
	}
	return out
}

// TestTrimSpansDoesNotCopy checks that trimming aliases the cache rather than copying it.
// A copy would be correct but would defeat the point: the cache exists to avoid moving
// key and value data around.
func TestTrimSpansDoesNotCopy(t *testing.T) {
	spans, data := makeSpans(4, 6)
	trimmed := trimSpans(spans, 3)
	if len(trimmed) != 1 {
		t.Fatalf("got %d spans, want 1", len(trimmed))
	}
	// The surviving span must alias the original backing array.
	before := data[len(data)-1]
	trimmed[0].data[trimmed[0].rows-1] = 99
	if data[len(data)-1] != 99 {
		t.Fatalf("trim copied instead of aliasing: backing array unchanged (was %v)", before)
	}
}

// TestSlidingWindowChangesAttention proves the window is actually applied, rather than
// configured and ignored. A window smaller than the context must change the output of a
// later position and leave earlier ones alone.
func TestSlidingWindowChangesAttention(t *testing.T) {
	c := defaultTiny()
	c.ctx = 32
	path := buildTinyWeights(t, c)

	full, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// Reload the same weights but with a two-position window, which is smaller than the
	// 16 positions below, so the window has to bite.
	full.W.Config.WindowLen = 2

	tokens := make([]int, 16)
	for i := range tokens {
		tokens[i] = i % 11
	}

	// Run the last position twice: once with the full context and once with the window.
	// Comparing the whole prefill is not the point; the window only changes positions
	// beyond its own reach.
	logitsAt := func(window int) []float32 {
		full.W.Config.WindowLen = window
		cache, err := NewKVCache(full.Config(), c.ctx)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		var out []float32
		for pos, tok := range tokens {
			lg, err := full.Forward(cache, tok, pos)
			if err != nil {
				t.Fatalf("Forward(pos=%d): %v", pos, err)
			}
			out = append([]float32(nil), lg...)
		}
		return out
	}

	wide := logitsAt(0)
	narrow := logitsAt(2)

	if len(wide) != len(narrow) {
		t.Fatalf("logit counts differ: %d vs %d", len(wide), len(narrow))
	}
	same := 0
	for i := range wide {
		if wide[i] == narrow[i] {
			same++
		}
	}
	if same == len(wide) {
		t.Fatal("a two-position window produced identical logits to full attention; the window is not applied")
	}
	for i := range wide {
		if math.IsNaN(float64(narrow[i])) {
			t.Fatalf("windowed logit[%d] is NaN", i)
		}
	}
}

// TestSlidingWindowLeavesEarlyPositionsAlone checks the other half of the property: a
// window that has not yet filled must not change anything, because every position so far
// is inside it. A window applied with an off-by-one would bite at position 0 and this
// would catch it.
func TestSlidingWindowLeavesEarlyPositionsAlone(t *testing.T) {
	c := defaultTiny()
	c.ctx = 32
	path := buildTinyWeights(t, c)

	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	logitsAt := func(window int) []float32 {
		m.W.Config.WindowLen = window
		cache, err := NewKVCache(m.Config(), c.ctx)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		var out []float32
		// Four positions with a window of 8: the window never applies.
		for pos := range 4 {
			lg, err := m.Forward(cache, pos%11, pos)
			if err != nil {
				t.Fatalf("Forward(pos=%d): %v", pos, err)
			}
			out = append(out, lg...)
		}
		return out
	}
	wide := logitsAt(0)
	roomy := logitsAt(8)
	if len(wide) != len(roomy) {
		t.Fatalf("logit counts differ: %d vs %d", len(wide), len(roomy))
	}
	for i := range wide {
		if wide[i] != roomy[i] {
			t.Fatalf("logit[%d] = %v with an 8-position window and %v unbounded; "+
				"a window larger than the context must not change anything", i, roomy[i], wide[i])
		}
	}
}

// TestMistralPromptWithinWindowMatchesLlama is the property a real Mistral decode depends
// on: while the prompt is shorter than the window, the model must behave exactly like
// Llama, because every position is still in range. A window applied with an off-by-one,
// or one that trimmed from the wrong end, would change short-prompt output and this would
// catch it without needing a checkpoint.
func TestMistralPromptWithinWindowMatchesLlama(t *testing.T) {
	c := defaultTiny()
	c.ctx = 32
	path := buildTinyWeights(t, c)

	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	run := func(window int) []float32 {
		m.W.Config.WindowLen = window
		cache, err := NewKVCache(m.Config(), c.ctx)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		var out []float32
		// Six positions, window of eight: everything stays in range.
		for pos := range 6 {
			lg, err := m.Forward(cache, pos%11, pos)
			if err != nil {
				t.Fatalf("Forward(pos=%d): %v", pos, err)
			}
			out = append(out, lg...)
		}
		return out
	}

	plain := run(0)
	mistral := run(8)
	if len(plain) != len(mistral) {
		t.Fatalf("logit counts differ: %d vs %d", len(plain), len(mistral))
	}
	for i := range plain {
		if plain[i] != mistral[i] {
			t.Fatalf("logit[%d] = %v with a window of 8, %v without; a window wider than "+
				"the prompt must not change anything", i, mistral[i], plain[i])
		}
	}

	// And a window narrower than the prompt must change something, or the mechanism is
	// not wired up at all.
	tight := run(2)
	differs := false
	for i := range plain {
		if plain[i] != tight[i] {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("a two-position window produced identical logits to no window")
	}
}

// TestSlidingWindowOnThePagedPath covers attentionCached's window trim.
//
// There are two attention implementations that each trim for a window: forward.go's attention
// serves Model.Forward and a contiguous cache, and attention.go's attentionCached serves
// PrefillWithPrefix through a PagedKVCache. TestSlidingWindowChangesAttention exercises the
// first; the second requires a paged pool with a prefix index and a prompt the index matches in
// full, because logitsForCached only runs when every position is already cached.
//
// Worth its own test rather than trusting the pair to stay in step: the two functions are
// near-identical copies, and a window applied on one path but not the other means the same
// prompt attends differently depending on whether --blocks is set.
func TestSlidingWindowOnThePagedPath(t *testing.T) {
	c := defaultTiny()
	c.ctx = 32
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	cfg.WindowLen = 2
	m.W.Config.WindowLen = cfg.WindowLen

	// Two blocks of two positions, so a four-token prompt fills them and is published.
	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: 2, Blocks: 4})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	pool.EnablePrefixCache()

	tokens := []int32{0, 1, 2, 3}
	// First pass fills the pool and remembers the prefix.
	first := pool.New()
	if _, err := m.PrefillWithPrefix(first, tokens, 0); err != nil {
		t.Fatalf("PrefillWithPrefix (fill): %v", err)
	}
	_ = first.Reset()

	// Second pass over the same prompt: the index matches every block, so the whole prompt is
	// cached and logitsForCached runs the last position through attentionCached.
	second := pool.New()
	defer func() { _ = second.Reset() }()
	logits, err := m.PrefillWithPrefix(second, tokens, 0)
	if err != nil {
		t.Fatalf("PrefillWithPrefix (reuse): %v", err)
	}
	if len(logits) != cfg.VocabSize {
		t.Fatalf("logits length = %d, want %d", len(logits), cfg.VocabSize)
	}
	if second.ReusedPositions() == 0 {
		t.Fatal("nothing was reused, so logitsForCached did not run")
	}
	// A windowed model must not produce NaN here: trimming to two positions leaves attention
	// with something to read, but an off-by-one in the trim would not.
	for i, v := range logits {
		if math.IsNaN(float64(v)) {
			t.Fatalf("windowed logit[%d] is NaN", i)
		}
	}
}

// TestSlidingWindowOnTheBatchPaths covers the two trims in batch.go.
//
// A window has to be applied in four places, one per attention implementation: forward.go's
// attention (covered by TestSlidingWindowChangesAttention), attention.go's attentionCached
// (covered by TestSlidingWindowOnThePagedPath), and batch.go's attentionBatch and its decode
// sibling, which serve Prefill and DecodeStep. The last two had no test with a window at all.
//
// The batch versions matter because the engine always goes through them: every server request
// and every bench run prefills and decodes in batches, so a window applied on the single-sequence
// paths but not here would leave the sliding-window behaviour exercised only by `orbix run`.
func TestSlidingWindowOnTheBatchPaths(t *testing.T) {
	c := defaultTiny()
	c.ctx = 32
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	// Two positions, and a prompt longer than that, so the trim has something to drop.
	m.W.Config.WindowLen = 2

	tokens := make([]int32, 8)
	for i := range tokens {
		tokens[i] = int32(i % 11)
	}

	cache, err := NewKVCache(m.Config(), c.ctx)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	logits, err := m.Prefill(cache, tokens, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if len(logits) != cfg.VocabSize {
		t.Fatalf("logits length = %d, want %d", len(logits), cfg.VocabSize)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) {
			t.Fatalf("batch-windowed logit[%d] is NaN", i)
		}
	}

	// Then decode two sequences in one batch, which is the other trim site.
	cache2, err := NewKVCache(m.Config(), c.ctx)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	if _, err := m.Prefill(cache2, tokens[:4], 0); err != nil {
		t.Fatalf("Prefill (second): %v", err)
	}
	next, err := m.DecodeStep([]Cache{cache, cache2}, []int32{1, 2})
	if err != nil {
		t.Fatalf("DecodeStep: %v", err)
	}
	// One vocabulary's worth of logits per sequence in the batch.
	if len(next) != 2*cfg.VocabSize {
		t.Fatalf("decode logits length = %d, want %d for two sequences", len(next), 2*cfg.VocabSize)
	}
	for i, v := range next {
		if math.IsNaN(float64(v)) {
			t.Fatalf("decode-windowed logit[%d] is NaN", i)
		}
	}
}

// TestAttendRefusesInconsistentArguments covers the five preconditions attend checks before it
// touches memory.
//
// attend is the inner loop of every decoder layer and takes slices from three different caches,
// so a caller passing a short score buffer, an output of the wrong width, or spans whose rows do
// not add up would otherwise get a forward pass that reads or writes out of bounds. The checks
// turn that into an error naming the mismatch.
//
// The span cases are the ones a refactor produces without touching a length: spans that each look
// valid but together cover one position too few. Enlarging a bound from != to < passes every
// existing test, because nothing in the suite ever calls attend with an argument it would refuse.
func TestAttendRefusesInconsistentArguments(t *testing.T) {
	const width = 2
	data := make([]float32, 8)
	// One span covering two positions, stride 4 so the rows are distinct slices.
	good := []span{{data: data, stride: 4, rows: 2, width: width}}
	q := make([]float32, width)
	out := make([]float32, width)
	scores := make([]float32, 2)

	// The happy path first, so a refusal below is about the argument rather than about attend.
	if err := attend(q, good, good, 2, scores, out, 0.5); err != nil {
		t.Fatalf("attend with consistent arguments: %v", err)
	}

	cases := []struct {
		name   string
		n      int
		keys   []span
		values []span
		scores []float32
		out    []float32
		want   string
	}{
		{
			name: "no positions", n: 0, keys: good, values: good,
			scores: scores, out: out, want: "attention over 0 positions",
		},
		{
			name: "negative positions", n: -1, keys: good, values: good,
			scores: scores, out: out, want: "attention over",
		},
		{
			name: "score buffer too short", n: 2, keys: good, values: good,
			scores: make([]float32, 1), out: out, want: "score buffer holds 1",
		},
		{
			name: "output narrower than the query", n: 2, keys: good, values: good,
			scores: scores, out: make([]float32, width-1), want: "attention output is 1 wide",
		},
		{
			name: "output wider than the query", n: 2, keys: good, values: good,
			scores: scores, out: make([]float32, width+1), want: "attention output is 3 wide",
		},
		{
			name: "keys cover too few positions", n: 2,
			keys: []span{{data: data, stride: 4, rows: 1, width: width}}, values: good,
			scores: scores, out: out, want: "keys cover 1 positions",
		},
		{
			name: "values cover too few positions", n: 2, keys: good,
			values: []span{{data: data, stride: 4, rows: 1, width: width}},
			scores: scores, out: out, want: "values cover 1 positions",
		},
		{
			// More rows than positions is the case a "<" comparison accepts: the extra rows
			// would be written to scores[n] and past it, which the n = 2 buffer does not have.
			name: "keys cover too many positions", n: 2,
			keys: []span{{data: data, stride: 4, rows: 2, width: width},
				{data: data, stride: 4, rows: 1, width: width}}, values: good,
			scores: scores, out: out, want: "keys cover 3 positions",
		},
		{
			name: "values cover too many positions", n: 2, keys: good,
			values: []span{{data: data, stride: 4, rows: 2, width: width},
				{data: data, stride: 4, rows: 1, width: width}},
			scores: scores, out: out, want: "values cover 3 positions",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := attend(q, c.keys, c.values, c.n, c.scores, c.out, 0.5)
			if err == nil {
				t.Fatalf("attend accepted %s", c.name)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not say %q", err, c.want)
			}
		})
	}
}
