package kernel

import (
	"math/rand"
	"runtime"
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
			b.ReportMetric(float64(r.parts(work, s.n)), "workers")
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
