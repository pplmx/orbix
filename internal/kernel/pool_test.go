package kernel

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"testing"
)

// A Runner must be a scheduling change, not a numerical one. These tests compare
// it against the serial reference with exact equality on purpose: float32
// addition is not associative, so "close enough" would hide a split that
// reassociated a reduction.

func TestRunnerMatVecIsBitIdentical(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	// Large enough to cross minParallelWork, which is where the code path
	// under test actually divides the rows.
	const n, k = 4096, 320
	a := randSlice(r, n*k)
	v := randSlice(r, k)

	want := make([]float32, n)
	if err := MatVec(want, a, v, n, k); err != nil {
		t.Fatalf("MatVec: %v", err)
	}

	for _, workers := range []int{1, 2, 3, 4, 8, 16, 64} {
		got := make([]float32, n)
		runner := NewRunner(workers)
		if err := runner.MatVec(got, a, v, n, k); err != nil {
			t.Fatalf("workers=%d: %v", workers, err)
		}
		for j := range want {
			if got[j] != want[j] {
				t.Fatalf("workers=%d: dst[%d] = %v, want %v", workers, j, got[j], want[j])
			}
		}
	}
}

func TestRunnerMatMulTransposedIsBitIdentical(t *testing.T) {
	r := rand.New(rand.NewSource(12))
	const m, n, k = 256, 512, 192
	a := randSlice(r, m*k)
	b := randSlice(r, n*k)

	want := make([]float32, m*n)
	if err := MatMulTransposed(want, a, b, m, n, k); err != nil {
		t.Fatalf("MatMulTransposed: %v", err)
	}
	for _, workers := range []int{1, 2, 5, 8, 32} {
		got := make([]float32, m*n)
		runner := NewRunner(workers)
		if err := runner.MatMulTransposed(got, a, b, m, n, k); err != nil {
			t.Fatalf("workers=%d: %v", workers, err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("workers=%d: dst[%d] = %v, want %v", workers, i, got[i], want[i])
			}
		}
	}
}

func TestRunnerHandlesRowCountsThatDoNotDivide(t *testing.T) {
	// Chunk boundaries that leave a short final chunk, and a row count smaller
	// than the worker count, are the two shapes an off-by-one would break.
	r := rand.New(rand.NewSource(13))
	for _, n := range []int{1, 2, 3, 7, 8, 9, 17} {
		const k = 1000
		a := randSlice(r, n*k)
		v := randSlice(r, k)
		want := make([]float32, n)
		if err := MatVec(want, a, v, n, k); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, n)
		if err := NewRunner(4).MatVec(got, a, v, n, k); err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		for j := range want {
			if got[j] != want[j] {
				t.Fatalf("n=%d: dst[%d] = %v, want %v", n, j, got[j], want[j])
			}
		}
	}
}

func TestNilRunnerIsSerial(t *testing.T) {
	// A nil Runner is the zero value: usable, one worker. Callers that never
	// configure concurrency must not have to check for nil.
	var runner *Runner
	if got := runner.Workers(); got != 1 {
		t.Errorf("nil Runner Workers() = %d, want 1", got)
	}
	a := []float32{1, 2, 3, 4}
	v := []float32{1, 1}
	got := make([]float32, 2)
	if err := runner.MatVec(got, a, v, 2, 2); err != nil {
		t.Fatalf("MatVec: %v", err)
	}
	if got[0] != 3 || got[1] != 7 {
		t.Errorf("dst = %v, want [3 7]", got)
	}
}

func TestRunnerRejectsShortBuffers(t *testing.T) {
	runner := NewRunner(4)
	if err := runner.MatVec(make([]float32, 2), make([]float32, 2), make([]float32, 2), 2, 2); err == nil {
		t.Error("Runner.MatVec accepted a short weight matrix")
	}
	if err := runner.MatVec(make([]float32, 2), make([]float32, 4), make([]float32, 1), 2, 2); err == nil {
		t.Error("Runner.MatVec accepted a short vector")
	}
	if err := runner.MatMulTransposed(make([]float32, 4), make([]float32, 4), make([]float32, 1), 2, 2, 2); err == nil {
		t.Error("Runner.MatMulTransposed accepted a short b")
	}
}

// TestRunnerEdgeCases covers the two branches in Runner that a workload with real
// shapes never reaches: a nil Runner, and a split asked for more parts than it has
// rows to divide.
//
// The nil case is the documented way to say "one worker", so it is a supported input
// rather than an accident, and a method that panicked on it would break every caller
// that leaves concurrency unset. The oversubscribed split is what happens when a
// caller sizes parts from a worker count instead of from the work, which is the normal
// case for a small decode step.
func TestRunnerEdgeCases(t *testing.T) {
	var nilRunner *Runner
	if got := nilRunner.Workers(); got != 1 {
		t.Errorf("nil Runner reports %d workers, want 1", got)
	}
	// The non-nil path reports what it was constructed with -- the half of the accessor
	// that actually answers a caller's question.
	if got := NewRunner(3).Workers(); got != 3 {
		t.Errorf("NewRunner(3).Workers() = %d, want 3", got)
	}
	// A nil Runner must still compute, not just report a count.
	dst := make([]float32, 2)
	if err := nilRunner.MatVec(dst, []float32{1, 0, 0, 1}, []float32{1, 2}, 2, 2); err != nil {
		t.Fatalf("nil Runner.MatVec: %v", err)
	}
	if dst[0] != 1 || dst[1] != 2 {
		t.Errorf("nil Runner.MatVec = %v, want [1 2]", dst)
	}

	// More parts than rows: 16 workers over a 2-row product must still finish and give
	// the serial answer, not deadlock or skip rows. Exercised on MatMulTransposed too,
	// because that is the one that splits columns -- a different call site, so the
	// "more parts than there is work" clamp is reached from both.
	busy := NewRunner(16)
	dst2 := make([]float32, 2)
	if err := busy.MatVec(dst2, []float32{1, 0, 0, 1}, []float32{3, 4}, 2, 2); err != nil {
		t.Fatalf("oversubscribed MatVec: %v", err)
	}
	if dst2[0] != 3 || dst2[1] != 4 {
		t.Errorf("oversubscribed MatVec = %v, want [3 4]", dst2)
	}
	// 1 row x 2 columns x 1 reduction with 16 workers: two columns, far fewer parts than
	// workers, so the column split clamps as well.
	dst3 := make([]float32, 2)
	if err := busy.MatMulTransposed(dst3, []float32{2}, []float32{3, 4}, 1, 2, 1); err != nil {
		t.Fatalf("oversubscribed MatMulTransposed: %v", err)
	}
	if dst3[0] != 6 || dst3[1] != 8 {
		t.Errorf("oversubscribed MatMulTransposed = %v, want [6 8]", dst3)
	}
}

// TestSplitRowsClampsMorePartsThanRows covers the clamp in splitRows, which is reachable
// with a legal shape rather than only as a guard: parts comes from parts(work), work is
// m*n*k, and splitOverRows picks the row split when m > n. A tall, narrow, deep product
// therefore asks for more row chunks than there are rows.
//
// Shape chosen from those rules: m=4, n=2, k=20000 gives work=160000, so 16 workers
// become parts=9, and splitOverRows(4, 2, 9) is true because m > n -- nine chunks of a
// four-row range. Without the clamp the chunk arithmetic would produce empty and
// out-of-range slices for five of them.
func TestSplitRowsClampsMorePartsThanRows(t *testing.T) {
	const m, n, k = 4, 2, 20000
	r := NewRunner(16)
	if parts := r.parts(m * n * k); parts <= m {
		t.Fatalf("parts(%d) = %d, need more than m=%d to reach the clamp", m*n*k, parts, m)
	}
	if !splitOverRows(m, n, r.parts(m*n*k)) {
		t.Fatal("this shape does not take the row split, so it does not reach splitRows")
	}

	dst := make([]float32, m*n)
	a := make([]float32, m*k)
	b := make([]float32, n*k)
	for i := range a {
		a[i] = 1
	}
	for i := range b {
		b[i] = 1
	}
	if err := r.MatMulTransposed(dst, a, b, m, n, k); err != nil {
		t.Fatalf("MatMulTransposed: %v", err)
	}
	// Every column of b is all ones over k terms, so each output is exactly k.
	for i, v := range dst {
		if v != float32(k) {
			t.Errorf("dst[%d] = %v, want %d: a clamped-but-wrong split would leave some rows short", i, v, k)
		}
	}
}

// BenchmarkRunnerCrossover reports the parallel and serial forms at several
// sizes. It is the evidence for minParallelWork: the parallel form loses below
// the threshold because spawning goroutines and joining them costs more than the
// arithmetic it splits.
func BenchmarkRunnerCrossover(b *testing.B) {
	shapes := []struct{ n, k int }{
		{288, 288},   // one attention projection of a small model
		{768, 288},   // one FFN projection of stories15M
		{32000, 288}, // the vocabulary projection
	}
	r := NewRunner(runtime.GOMAXPROCS(0))
	v := make([]float32, 32000)
	for i := range v {
		v[i] = float32(i%7) / 7
	}
	for _, s := range shapes {
		a := make([]float32, s.n*s.k)
		for i := range a {
			a[i] = float32(i%13) / 13
		}
		dst := make([]float32, s.n)
		work := s.n * s.k

		b.Run(sizeName(s.n, s.k, "serial"), func(b *testing.B) {
			b.SetBytes(int64(work) * 8)
			for range b.N {
				if err := MatVec(dst, a, v[:s.k], s.n, s.k); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(sizeName(s.n, s.k, "parallel"), func(b *testing.B) {
			b.SetBytes(int64(work) * 8)
			for range b.N {
				if err := r.MatVec(dst, a, v[:s.k], s.n, s.k); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(sizeName(s.n, s.k, "workers"), func(b *testing.B) {
			b.ReportMetric(float64(r.parts(work)), "workers")
			b.Skip("report-only: the split each size receives")
		})
	}
}

func sizeName(n, k int, variant string) string {
	return variant + "/" + itoa(n) + "x" + itoa(k)
}

// itoa avoids pulling strconv into a benchmark helper for three numbers.
func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [8]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

// TestRunnerSplitsShortAndWideProducts is the regression for the split shape itself. An
// m=1 product against a 32000-wide vocabulary has one row, so a split over rows alone
// would run entirely on one goroutine -- the exact case a batched decode depends on.
//
// The value comparison below cannot catch that on its own: a serial kernel returns the
// same numbers as a parallel one, which is the whole point of the split being over
// independent elements. It passed for a release while single-stream decode had quietly
// become 2.2x slower, because splitting over rows left m = 1 with one worker. The
// predicate assertion in TestMatMulTransposedSplitsTheAxisThatHasWork is what closes that
// gap; this test keeps the values honest.
func TestRunnerSplitsShortAndWideProducts(t *testing.T) {
	r := rand.New(rand.NewSource(21))
	const n, k = 32000, 288
	for _, m := range []int{1, 2, 5} {
		a := randSlice(r, m*k)
		b := randSlice(r, n*k)

		want := make([]float32, m*n)
		if err := MatMulTransposed(want, a, b, m, n, k); err != nil {
			t.Fatalf("MatMulTransposed: %v", err)
		}
		got := make([]float32, m*n)
		if err := NewRunner(16).MatMulTransposed(got, a, b, m, n, k); err != nil {
			t.Fatalf("Runner.MatMulTransposed(m=%d): %v", m, err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("m=%d: dst[%d] = %v, want %v", m, i, got[i], want[i])
			}
		}
	}
}

// TestMatMulTransposedSplitsTheAxisThatHasWork pins which axis the product is split over,
// not which numbers come out of it.
//
// This is the regression for a 2.2x slowdown in single-stream decode that shipped and
// stayed: MatMulTransposed was split over the rows of a, a decode step has exactly one of
// those, and so the vocabulary projection -- the largest single product in a token -- ran
// on one goroutine. No value changed, so comparing against the serial kernel could not see
// it, and the benchmark that would have caught it does not gate.
//
// The decision is now a predicate, splitOverRows, and this pins it at the shapes that
// matter. m = 1 is the case that regressed; a tall product is the case a row split is still
// right for, and it is here so the test cannot be satisfied by always answering "columns".
func TestMatMulTransposedSplitsTheAxisThatHasWork(t *testing.T) {
	const parts = 64
	cases := []struct {
		name     string
		m, n     int
		wantRows bool
		why      string
	}{
		{
			name: "single-stream decode", m: 1, n: 32000, wantRows: false,
			why: "one activation row against the whole vocabulary; a row split would be one worker",
		},
		{
			name: "small batch decode", m: 4, n: 32000, wantRows: false,
			why: "four rows against 32000 columns; columns have far more to divide",
		},
		{
			name: "batched prefill", m: 2001, n: 2048, wantRows: false,
			why: "columns still outnumber rows, and n is the larger axis",
		},
		{
			name: "tall and narrow", m: 4096, n: 64, wantRows: true,
			why: "64 columns cannot fill 64 workers, 4096 rows can",
		},
		{
			name: "square", m: 2048, n: 2048, wantRows: false,
			why: "neither axis has more to divide, so the column default stands; a square " +
				"projection is an ordinary shape (attention output to output)",
		},
		{
			name: "serial product", m: 1, n: 8, wantRows: false,
			why: "below the parallel threshold, so the axis does not arise",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := splitOverRows(c.m, c.n, parts)
			if got != c.wantRows {
				t.Errorf("splitOverRows(m=%d, n=%d, parts=%d) = %v, want %v: %s",
					c.m, c.n, parts, got, c.wantRows, c.why)
			}
		})
	}

	// One worker means no split at all, whichever axis is chosen; asserting it keeps the
	// predicate from claiming a row split the Runner would never perform.
	if splitOverRows(4096, 64, 1) {
		t.Error("splitOverRows reported a row split with one part, but a single part is not a split")
	}
}

// TestRunnerMatMulTransposedIsExactUnderBothSplits checks that the two axes agree.
//
// The axis is chosen inside MatMulTransposed, so both branches need a shape that reaches
// them: m > n for the row branch, m <= n for the column branch. Both must equal the serial
// kernel exactly, because the split moves work between workers and never reorders the p
// loop inside an element.
func TestRunnerMatMulTransposedIsExactUnderBothSplits(t *testing.T) {
	r := rand.New(rand.NewSource(77))
	// parts() needs enough work to fan out at all, which is what makes the axis matter.
	shapes := []struct {
		name string
		m    int
		n    int
		k    int
	}{
		{"column branch: one row, wide", 1, 2048, 512},
		{"column branch: few rows, wide", 3, 1024, 512},
		{"row branch: many rows, narrow", 4096, 64, 128},
		{"row branch: tall, very narrow", 2048, 8, 512},
	}
	for _, sh := range shapes {
		t.Run(sh.name, func(t *testing.T) {
			a := randSlice(r, sh.m*sh.k)
			b := randSlice(r, sh.n*sh.k)
			want := make([]float32, sh.m*sh.n)
			if err := MatMulTransposed(want, a, b, sh.m, sh.n, sh.k); err != nil {
				t.Fatalf("MatMulTransposed: %v", err)
			}
			for _, workers := range []int{2, 7, 64} {
				got := make([]float32, sh.m*sh.n)
				if err := NewRunner(workers).MatMulTransposed(got, a, b, sh.m, sh.n, sh.k); err != nil {
					t.Fatalf("%d workers: %v", workers, err)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("%d workers: dst[%d] = %v, want %v", workers, i, got[i], want[i])
					}
				}
			}
		})
	}
}

// BenchmarkBatchAmortizes is the evidence that batched decode is worth building:
// it compares B matvecs against one m=B product at the vocabulary projection,
// which is where a decode step spends most of its time.
func BenchmarkBatchAmortizes(b *testing.B) {
	runner := NewRunner(runtime.GOMAXPROCS(0))
	const vocab, hidden = 32000, 288
	w := make([]float32, vocab*hidden)
	for i := range w {
		w[i] = float32(i%13)/13 - 0.5
	}
	for _, batch := range []int{1, 4, 8, 16, 32} {
		x := make([]float32, batch*hidden)
		for i := range x {
			x[i] = float32(i%7)/7 - 0.5
		}
		dst := make([]float32, batch*vocab)
		row := make([]float32, vocab)

		b.Run("batched/"+itoa(batch), func(b *testing.B) {
			b.SetBytes(int64(batch) * vocab * hidden * 8 / int64(batch))
			for range b.N {
				if err := runner.MatMulTransposed(dst, x, w, batch, vocab, hidden); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run("percall/"+itoa(batch), func(b *testing.B) {
			b.SetBytes(int64(batch) * vocab * hidden * 8 / int64(batch))
			for range b.N {
				for s := range batch {
					if err := runner.MatVec(row, w, x[s*hidden:(s+1)*hidden], vocab, hidden); err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}

// TestParallelCoversEveryIndexExactlyOnce covers the contract Parallel exists to provide.
//
// It is the concurrency primitive behind attention, where each position writes its own slice of the
// output and reads only the cache. Two properties make that safe, and neither had a test: every
// index runs exactly once, and the slots handed to workers stay distinct so two workers cannot
// write the same scratch array. A chunking bug shows up as a skipped index or a doubled one, both of
// which produce wrong output rather than a crash.
//
// The grid covers the cases the ceil division has to get right: count below the worker count (fewer
// parts than indices), count not divisible by the parts, and the degenerate zero.
func TestParallelCoversEveryIndexExactlyOnce(t *testing.T) {
	for _, workers := range []int{0, 1, 2, 3, 8} {
		for _, count := range []int{0, 1, 2, 5, 7, 16} {
			t.Run(fmt.Sprintf("workers=%d/count=%d", workers, count), func(t *testing.T) {
				r := NewRunner(workers)
				var mu sync.Mutex
				seen := make(map[int]int, count)
				slots := make(map[int]struct{})
				r.Parallel(count, func(i, slot int) {
					mu.Lock()
					seen[i]++
					slots[slot] = struct{}{}
					mu.Unlock()
				})

				for i := range count {
					switch seen[i] {
					case 1:
					case 0:
						t.Errorf("index %d never visited", i)
					default:
						t.Errorf("index %d visited %d times", i, seen[i])
					}
				}
				// A slot beyond the worker count means the chunking produced more parts than
				// workers, which would let two of them share scratch.
				maxSlots := min(max(workers, 1), count)
				if len(slots) > maxSlots {
					t.Errorf("%d distinct slots for %d workers and %d indices", len(slots), workers, count)
				}
			})
		}
	}
}

// TestPartsPinsTheThresholds checks the two tuning constants through the predicate that
// applies them.
//
// parts decides how many goroutines a product gets, from a measured crossover: below
// minParallelWork the spawn-and-join costs more than the arithmetic, and each worker beyond
// the first needs workPerWorker of the product to pay for itself. Both numbers are records of
// a measurement, so the test asserts the shape of the decision rather than re-deriving the
// constants from a benchmark that would not gate.
//
// The boundary case is the point. `work < minParallelWork` and `work <= minParallelWork`
// agree everywhere except exactly at the threshold, and `<=` passed the whole kernel, llama
// and engine suites -- a one-bit change to a constant that is otherwise only exercised well
// away from it on either side.
func TestPartsPinsTheThresholds(t *testing.T) {
	r := NewRunner(64)

	cases := []struct {
		name string
		work int
		want int
		why  string
	}{
		{"nothing", 0, 1, "no product at all stays serial"},
		{"one below the crossover", minParallelWork - 1, 1, "spawning would cost more than the work"},
		{"exactly the crossover", minParallelWork, 2, "at the threshold the parallel form breaks even, so it runs"},
		{"one worker's worth", workPerWorker, 1, "one worker's worth is not two workers' worth"},
		{"two workers' worth", 2 * workPerWorker, 2, "each worker needs workPerWorker of the product"},
		{"many workers' worth", 64 * workPerWorker, 64, "bounded by the configured worker count"},
		{"more than every worker", 1 << 30, 64, "the cap is the runner's own count"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := r.parts(c.work); got != c.want {
				t.Errorf("parts(%d) = %d, want %d: %s", c.work, got, c.want, c.why)
			}
		})
	}

	// A nil Runner and a one-worker Runner both mean serial, and neither may divide by a
	// worker count it does not have.
	var nilRunner *Runner
	if got := nilRunner.parts(1 << 30); got != 1 {
		t.Errorf("nil Runner parts = %d, want 1", got)
	}
	if got := NewRunner(1).parts(1 << 30); got != 1 {
		t.Errorf("one-worker parts = %d, want 1", got)
	}
	// Zero is the value callers pass to mean "decide for me", and this type does not decide:
	// it takes the number it was given. A Runner built with 0 therefore runs serial here,
	// while Workers() reports 0 -- so a caller expanding "0 means GOMAXPROCS" has to do it
	// before construction. llama.workers does exactly that, which is what keeps the two from
	// disagreeing in a served model.
	//
	// These two assertions record that contract rather than guarding it: changing the
	// condition above from "<= 1" to "< 1" lets 0 through to min(0, work/workPerWorker) = 0,
	// and split's own "parts <= 1" then runs the same serial branch. The behaviour is
	// unchanged, so removing the guard here is an equivalent mutant and no test can fail on
	// it. The value of these lines is that the next reader sees the decision was deliberate.
	if got := NewRunner(0).parts(1 << 30); got != 1 {
		t.Errorf("zero-worker parts = %d, want 1", got)
	}
	if got := NewRunner(0).Workers(); got != 0 {
		t.Errorf("NewRunner(0).Workers() = %d, want 0: the type reports what it was given", got)
	}
}
