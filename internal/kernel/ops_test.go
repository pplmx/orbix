package kernel

import (
	"math"
	"math/rand"
	"strconv"
	"strings"
	"testing"
)

// The reference values in these tests are computed by straightforward,
// independently written loops in the test itself rather than by copying
// numbers from elsewhere. If the kernel and the naive version disagree, one of
// them is wrong and the difference tells you which.

func randSlice(r *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = r.Float32()*2 - 1
	}
	return out
}

func TestMatMulAgainstNaive(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	const m, n, k = 5, 7, 3
	a := randSlice(r, m*k)
	b := randSlice(r, k*n)
	dst := make([]float32, m*n)

	if err := MatMul(dst, a, b, m, n, k); err != nil {
		t.Fatalf("MatMul: %v", err)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var want float64
			for p := 0; p < k; p++ {
				want += float64(a[i*k+p]) * float64(b[p*n+j])
			}
			if got := float64(dst[i*n+j]); math.Abs(got-want) > 1e-6 {
				t.Errorf("dst[%d,%d] = %v, want %v", i, j, got, want)
			}
		}
	}
}

func TestMatMulMatchesTransposed(t *testing.T) {
	// a @ b with b as k x n and a @ bᵀ with bᵀ as n x k must agree. This pins
	// the two functions to the same mathematical definition even though they
	// index their inputs differently.
	r := rand.New(rand.NewSource(2))
	const m, n, k = 4, 6, 5
	a := randSlice(r, m*k)
	b := randSlice(r, k*n)
	bT := make([]float32, n*k)
	for p := 0; p < k; p++ {
		for j := 0; j < n; j++ {
			bT[j*k+p] = b[p*n+j]
		}
	}

	want := make([]float32, m*n)
	if err := MatMul(want, a, b, m, n, k); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, m*n)
	if err := MatMulTransposed(got, a, bT, m, n, k); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Errorf("[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestMatMulAssignsNotAccumulates(t *testing.T) {
	// dst is overwritten, not added to. A stale value must not survive.
	a := []float32{1, 2}
	b := []float32{3, 4}
	dst := []float32{99, 99}
	if err := MatMul(dst, a, b, 1, 2, 1); err != nil {
		t.Fatal(err)
	}
	if dst[0] != 3 || dst[1] != 4 {
		t.Errorf("dst = %v, want [3 4]", dst)
	}
}

func TestMatVec(t *testing.T) {
	// A single activation against a stack of rows: the decode-time path.
	r := rand.New(rand.NewSource(3))
	const n, k = 6, 4
	a := randSlice(r, n*k)
	v := randSlice(r, k)
	dst := make([]float32, n)
	if err := MatVec(dst, a, v, n, k); err != nil {
		t.Fatalf("MatVec: %v", err)
	}
	for j := 0; j < n; j++ {
		var want float64
		for p := 0; p < k; p++ {
			want += float64(a[j*k+p]) * float64(v[p])
		}
		if math.Abs(float64(dst[j])-want) > 1e-6 {
			t.Errorf("dst[%d] = %v, want %v", j, dst[j], want)
		}
	}
}

func TestMatMulRejectsShortBuffers(t *testing.T) {
	tests := []struct {
		name string
		dst  int
		a    int
		b    int
	}{
		{"short dst", 3, 6, 6},
		{"short a", 6, 3, 6},
		{"short b", 6, 6, 3},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := MatMul(make([]float32, tc.dst), make([]float32, tc.a), make([]float32, tc.b), 2, 3, 2)
			if err == nil {
				t.Fatal("MatMul succeeded with a short buffer")
			}
		})
	}
}

// TestMatVecRejectsShortBuffers covers the serial MatVec's validation, which no test
// reached: every short-buffer case above is MatMul, and the Runner tests cover the
// parallel entry point. The serial function is what the Runner falls back to below
// minParallelWork, so a missing check there is reachable by any small decode step.
func TestMatVecRejectsShortBuffers(t *testing.T) {
	// 2x3 weights, a 3-vector, 2 outputs: the smallest shape that has every buffer
	// distinct, so a short one is unambiguous.
	const n, k = 2, 3
	tests := []struct {
		name      string
		dst, a, v int
		wantMsg   string
	}{
		{"short dst", 1, n * k, k, "dst has"},
		{"short weight matrix", n, n*k - 1, k, "a has"},
		{"short vector", n, n * k, k - 1, "v has"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := MatVec(make([]float32, tc.dst), make([]float32, tc.a), make([]float32, tc.v), n, k)
			if err == nil {
				t.Fatal("MatVec succeeded with a short buffer")
			}
			// The message has to name the buffer: with four of them, "invalid shape"
			// would leave the caller guessing which slice to resize.
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Errorf("error %q does not name %q", err, tc.wantMsg)
			}
		})
	}
}

// TestMatMulTransposedRejectsShortBuffers covers the serial MatMulTransposed's
// validation. Its b buffer is the transposed weight matrix, the largest slice in a
// vocab projection, and a short one would otherwise index past the end.
func TestMatMulTransposedRejectsShortBuffers(t *testing.T) {
	const m, n, k = 2, 3, 2
	tests := []struct {
		name      string
		dst, a, b int
		wantMsg   string
	}{
		{"short dst", m*n - 1, m * k, n * k, "dst has"},
		{"short a", m * n, m*k - 1, n * k, "a has"},
		{"short b", m * n, m * k, n*k - 1, "b has"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := MatMulTransposed(make([]float32, tc.dst), make([]float32, tc.a), make([]float32, tc.b), m, n, k)
			if err == nil {
				t.Fatal("MatMulTransposed succeeded with a short buffer")
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Errorf("error %q does not name %q", err, tc.wantMsg)
			}
		})
	}
}

// TestRMSNormRejectsShortBuffers covers RMSNorm's two length checks and its empty
// input. The empty case is not an error -- a zero-width row normalizes to nothing --
// but it must return before dividing by n, which is the reason it is a separate branch
// rather than a condition on the loop.
func TestRMSNormRejectsShortBuffers(t *testing.T) {
	const n = 4
	src := make([]float32, n)
	if err := RMSNorm(make([]float32, n-1), src, make([]float32, n), 1e-6); err == nil {
		t.Error("RMSNorm accepted a short destination")
	} else if !strings.Contains(err.Error(), "dst has") {
		t.Errorf("error %q does not name the destination", err)
	}
	if err := RMSNorm(make([]float32, n), src, make([]float32, n-1), 1e-6); err == nil {
		t.Error("RMSNorm accepted a short weight")
	} else if !strings.Contains(err.Error(), "weight has") {
		t.Errorf("error %q does not name the weight", err)
	}
	// An empty row must succeed without touching anything or dividing by zero.
	if err := RMSNorm(nil, nil, nil, 1e-6); err != nil {
		t.Errorf("RMSNorm on an empty row: %v", err)
	}
}

// TestKernelsRejectNegativeDimensions covers the guard that runs before any length
// check. A negative dimension makes n*k negative, so the length checks all pass and the
// loop bounds are empty -- the call would silently do nothing and return success, which
// is the failure mode a caller would never notice.
func TestKernelsRejectNegativeDimensions(t *testing.T) {
	// Buffers long enough that only the sign check can reject these.
	buf := make([]float32, 16)
	cases := []struct {
		name string
		err  error
	}{
		{"MatVec n<0", MatVec(buf, buf, buf, -1, 2)},
		{"MatVec k<0", MatVec(buf, buf, buf, 2, -1)},
		{"MatMul m<0", MatMul(buf, buf, buf, -1, 2, 2)},
		{"MatMul n<0", MatMul(buf, buf, buf, 2, -1, 2)},
		{"MatMul k<0", MatMul(buf, buf, buf, 2, 2, -1)},
		{"MatMulTransposed m<0", MatMulTransposed(buf, buf, buf, -1, 2, 2)},
		{"MatMulTransposed n<0", MatMulTransposed(buf, buf, buf, 2, -1, 2)},
		{"MatMulTransposed k<0", MatMulTransposed(buf, buf, buf, 2, 2, -1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.err == nil {
				t.Fatal("negative dimension was accepted; the call did nothing and reported success")
			}
			if !strings.Contains(tc.err.Error(), "negative") {
				t.Errorf("error %q does not say the dimension is negative", tc.err)
			}
		})
	}
}

func TestRMSNorm(t *testing.T) {
	src := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	const eps = 1e-6
	dst := make([]float32, 4)
	if err := RMSNorm(dst, src, weight, eps); err != nil {
		t.Fatalf("RMSNorm: %v", err)
	}

	// Independent computation of the same definition.
	var sum float64
	for _, v := range src {
		sum += float64(v) * float64(v)
	}
	rms := math.Sqrt(sum/4 + eps)
	for i := range src {
		want := float64(src[i]) / rms
		if math.Abs(float64(dst[i])-want) > 1e-6 {
			t.Errorf("dst[%d] = %v, want %v", i, dst[i], want)
		}
	}

	// The result's root mean square should be about 1 before weighting.
	var check float64
	for _, v := range dst {
		check += float64(v) * float64(v)
	}
	if got := math.Sqrt(check / 4); math.Abs(got-1) > 1e-3 {
		t.Errorf("rms of the normalized row = %v, want ~1", got)
	}
}

func TestRMSNormAppliesWeight(t *testing.T) {
	src := []float32{1, 1, 1, 1}
	weight := []float32{2, 2, 2, 2}
	dst := make([]float32, 4)
	if err := RMSNorm(dst, src, weight, 0); err != nil {
		t.Fatal(err)
	}
	// All-equal input normalizes to 1, so the result is exactly the weight.
	for i := range dst {
		if math.Abs(float64(dst[i])-2) > 1e-6 {
			t.Errorf("dst[%d] = %v, want 2", i, dst[i])
		}
	}
}

func TestRMSNormAliasesInput(t *testing.T) {
	// In-place use is allowed and must give the same answer as the copying
	// form, since every layer would otherwise need a scratch buffer.
	r := rand.New(rand.NewSource(4))
	src := randSlice(r, 8)
	weight := randSlice(r, 8)

	want := make([]float32, 8)
	if err := RMSNorm(want, src, weight, 1e-5); err != nil {
		t.Fatal(err)
	}
	inPlace := append([]float32{}, src...)
	if err := RMSNorm(inPlace, inPlace, weight, 1e-5); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if inPlace[i] != want[i] {
			t.Errorf("[%d] = %v, want %v", i, inPlace[i], want[i])
		}
	}
}

func TestRMSNormZeroRow(t *testing.T) {
	// An all-zero row must produce zeros rather than NaNs: eps is what makes
	// this well-defined.
	dst := make([]float32, 4)
	if err := RMSNorm(dst, make([]float32, 4), []float32{1, 1, 1, 1}, 1e-5); err != nil {
		t.Fatal(err)
	}
	for i, v := range dst {
		if v != 0 {
			t.Errorf("dst[%d] = %v, want 0", i, v)
		}
	}
}

func TestSoftmax(t *testing.T) {
	scores := []float32{1, 2, 3}
	Softmax(scores)
	// Independent computation.
	var want []float64
	var sum float64
	for _, v := range []float64{1, 2, 3} {
		e := math.Exp(v - 3)
		want = append(want, e)
		sum += e
	}
	var total float64
	for i := range scores {
		if math.Abs(float64(scores[i])-want[i]/sum) > 1e-6 {
			t.Errorf("[%d] = %v, want %v", i, scores[i], want[i]/sum)
		}
		total += float64(scores[i])
	}
	if math.Abs(total-1) > 1e-6 {
		t.Errorf("sum = %v, want 1", total)
	}
}

// TestSoftmaxEmpty covers the empty slice, which is a legal input: a batch with no rows
// reaches the kernel as a zero-length slice rather than being special-cased by callers.
// The alternative is slices.Max panicking on an empty slice, so the early return is
// load-bearing even though the arithmetic below it never runs.
func TestSoftmaxEmpty(t *testing.T) {
	Softmax(nil)
	Softmax([]float32{})
}

// TestMatMulZeroTermsDoNotChangeTheAnswer covers the sparse shortcut in MatMul's inner
// loop, which skips a row of b when the matching element of a is zero. That branch is
// for the mixture-of-experts path, where an unselected expert contributes an all-zero
// activation row.
//
// What this test does and does not prove is worth being precise about. It proves the
// shortcut is invisible: a zero term among nonzero ones yields the same product as the
// naive triple loop. It cannot prove the skip happens, because skipping a zero and
// multiplying by it are, by definition, the same computation -- removing the branch
// leaves this test passing, which was checked. The branch's own justification is that
// it is a speed optimization whose result must not change, and that is what is asserted
// here; the coverage number is what records that the line is reached.
func TestMatMulZeroTermsDoNotChangeTheAnswer(t *testing.T) {
	// a is 2x3 with a zero in the middle of row 0; b is 3x2.
	a := []float32{1, 0, 2}
	b := []float32{
		3, 4, // row 0 of b, multiplied by a[0]=1
		5, 6, // row 1, skipped because a[1]=0
		7, 8, // row 2, multiplied by a[2]=2
	}
	dst := make([]float32, 2)
	if err := MatMul(dst, a, b, 1, 2, 3); err != nil {
		t.Fatalf("MatMul: %v", err)
	}
	// Independent arithmetic: [1*3 + 0*5 + 2*7, 1*4 + 0*6 + 2*8] = [17, 20].
	want := []float32{3 + 2*7, 4 + 2*8}
	for i := range want {
		if dst[i] != want[i] {
			t.Errorf("dst[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

// TestMatMulAccumulatesFromZero covers the row-clearing loop: MatMul assigns its output
// rather than accumulating into whatever the destination held, so a destination
// pre-filled with garbage must come back with the product alone.
func TestMatMulAccumulatesFromZero(t *testing.T) {
	a := []float32{1, 2}
	b := []float32{3, 4}
	dst := []float32{1000, -1000}
	if err := MatMul(dst, a, b, 1, 1, 2); err != nil {
		t.Fatalf("MatMul: %v", err)
	}
	// 1*3 + 2*4 = 11, with neither pre-existing value surviving.
	if dst[0] != 11 {
		t.Errorf("dst[0] = %v, want 11: MatMul added to the old value instead of assigning", dst[0])
	}
}

func TestSoftmaxNoOverflow(t *testing.T) {
	// Logits around 1000 would make exp overflow if the maximum were not
	// subtracted first.
	scores := []float32{1000, 1001, 1002}
	Softmax(scores)
	var sum float64
	for _, v := range scores {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("overflowed: %v", scores)
		}
		sum += float64(v)
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Errorf("sum = %v, want 1", sum)
	}
}

func TestSoftmaxUniformOnEqualInput(t *testing.T) {
	scores := []float32{5, 5, 5, 5}
	Softmax(scores)
	for i, v := range scores {
		if math.Abs(float64(v)-0.25) > 1e-6 {
			t.Errorf("[%d] = %v, want 0.25", i, v)
		}
	}
}

func TestSiLU(t *testing.T) {
	// silu(0) = 0, silu(large positive) ~ x, silu(large negative) ~ 0.
	x := []float32{0, 10, -10}
	SiLU(x)
	if math.Abs(float64(x[0])) > 1e-6 {
		t.Errorf("silu(0) = %v, want 0", x[0])
	}
	if math.Abs(float64(x[1])-float64(10/(1+math.Exp(-10)))) > 1e-4 {
		t.Errorf("silu(10) = %v, want %v", x[1], 10/(1+math.Exp(-10)))
	}
	if x[2] < -1e-3 || math.IsNaN(float64(x[2])) {
		t.Errorf("silu(-10) = %v, want ~0", x[2])
	}

	// A very negative input must saturate to zero, not overflow.
	extreme := []float32{-1000}
	SiLU(extreme)
	if extreme[0] != 0 {
		t.Errorf("silu(-1000) = %v, want 0", extreme[0])
	}
}

func TestAddMulDot(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}

	sum := append([]float32{}, a...)
	if err := AddInPlace(sum, b); err != nil {
		t.Fatal(err)
	}
	if sum[0] != 5 || sum[2] != 9 {
		t.Errorf("a + b = %v", sum)
	}

	prod := append([]float32{}, a...)
	if err := MulInPlace(prod, b); err != nil {
		t.Fatal(err)
	}
	if prod[0] != 4 || prod[2] != 18 {
		t.Errorf("a * b = %v", prod)
	}

	got, err := Dot(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if want := float32(4 + 10 + 18); got != want {
		t.Errorf("Dot = %v, want %v", got, want)
	}

	if err := AddInPlace(a, []float32{1}); err == nil {
		t.Error("AddInPlace with mismatched lengths succeeded")
	}
	if err := MulInPlace(a, []float32{1}); err == nil {
		t.Error("MulInPlace with mismatched lengths succeeded")
	}
	if _, err := Dot(a, []float32{1}); err == nil {
		t.Error("Dot with mismatched lengths succeeded")
	}
}

func TestZeroDimensionMatMul(t *testing.T) {
	// A 0x0 product is legal and a no-op; it must not panic on an empty slice.
	if err := MatMul(nil, nil, nil, 0, 0, 0); err != nil {
		t.Errorf("MatMul(empty) = %v", err)
	}
}

func BenchmarkMatMul(b *testing.B) {
	// A 4096 x 4096 square product, the shape of a 7B model's attention
	// projection.
	const m, n, k = 256, 4096, 4096
	a := make([]float32, m*k)
	c := make([]float32, k*n)
	for i := range a {
		a[i] = float32(i%17) / 17
	}
	for i := range c {
		c[i] = float32(i%13) / 13
	}
	dst := make([]float32, m*n)
	b.SetBytes(int64(m*n*k) * 8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := MatMul(dst, a, c, m, n, k); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVec(b *testing.B) {
	const n, k = 4096, 4096
	a := make([]float32, n*k)
	v := make([]float32, k)
	for i := range a {
		a[i] = float32(i%17) / 17
	}
	dst := make([]float32, n)
	b.SetBytes(int64(n*k) * 8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := MatVec(dst, a, v, n, k); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRMSNorm(b *testing.B) {
	const n = 4096
	src := make([]float32, n)
	w := make([]float32, n)
	dst := make([]float32, n)
	for i := range src {
		src[i] = float32(i%19) / 19
		w[i] = 1
	}
	b.SetBytes(int64(n) * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := RMSNorm(dst, src, w, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

// TestMatMulTransposedRowTilingIsExact pins the property the row tiling in
// matMulTransposedCols depends on: the order output rows are visited changes, but each
// element is still the same p loop over the same two rows, so the result must be
// bit-identical to a naive element-at-a-time multiply.
//
// The tiling exists for memory traffic -- a row of b is read once per tile instead of
// once per output row -- and it would be a poor trade if it changed a single value.
func TestMatMulTransposedRowTilingIsExact(t *testing.T) {
	// Sizes chosen so a row tile boundary falls inside the range: m is not a multiple of
	// the tile, and the last tile is partial.
	const m, n, k = 19, 7, 5
	a := make([]float32, m*k)
	b := make([]float32, n*k)
	for i := range a {
		a[i] = float32(i%13-6) / 11
	}
	for i := range b {
		b[i] = float32(i%9-4) / 7
	}

	got := make([]float32, m*n)
	if err := MatMulTransposed(got, a, b, m, n, k); err != nil {
		t.Fatalf("MatMulTransposed: %v", err)
	}

	// Naive reference: one element at a time, same p order.
	want := make([]float32, m*n)
	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += a[i*k+p] * b[j*k+p]
			}
			want[i*n+j] = sum
		}
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("element %d = %v, want %v (row tiling changed a value)", i, got[i], want[i])
		}
	}
}

// TestMatMulTransposedMatchesRowRange checks that the row-tiled serial path and the
// row-split parallel path agree exactly, which is what makes the split safe.
func TestMatMulTransposedMatchesRowRange(t *testing.T) {
	const m, n, k = 33, 11, 9
	a := make([]float32, m*k)
	b := make([]float32, n*k)
	for i := range a {
		a[i] = float32(i%17-8) / 13
	}
	for i := range b {
		b[i] = float32(i%7-3) / 9
	}

	serial := make([]float32, m*n)
	if err := MatMulTransposed(serial, a, b, m, n, k); err != nil {
		t.Fatalf("MatMulTransposed: %v", err)
	}
	for _, workers := range []int{2, 5, 16} {
		parallel := make([]float32, m*n)
		r := NewRunner(workers)
		if err := r.MatMulTransposed(parallel, a, b, m, n, k); err != nil {
			t.Fatalf("MatMulTransposed(%d workers): %v", workers, err)
		}
		for i := range serial {
			if parallel[i] != serial[i] {
				t.Fatalf("workers=%d element %d = %v, serial %v", workers, i, parallel[i], serial[i])
			}
		}
	}
}

// TestSoftmaxPropagatesNonFinite pins what the reduction does with a NaN or an infinity.
//
// A NaN logit makes every output NaN, and that is the correct result rather than a bug: the
// normaliser is a sum over the row, so one poisoned term poisons the total and the whole
// distribution becomes meaningless. The alternative -- returning a distribution that looks
// valid -- would turn a corrupt weight into plausible text, which is the failure mode worth
// avoiding. Greedy decoding reads the logits without going through this function, so it can
// still pick a token when Softmax cannot.
//
// The test exists because nothing pinned it: the behaviour is reachable only from a damaged
// checkpoint, and "the row comes back all NaN" is the observation that identifies one.
func TestSoftmaxPropagatesNonFinite(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))

	for _, c := range []struct {
		name   string
		in     []float32
		allNaN bool
	}{
		{"one NaN poisons the row", []float32{1, nan, 3}, true},
		{"every value NaN", []float32{nan, nan}, true},
		{"+Inf as the maximum", []float32{1, inf, 3}, true},
		{"-Inf is ordinary", []float32{0, float32(math.Inf(-1))}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			scores := append([]float32(nil), c.in...)
			Softmax(scores)
			var sawNaN bool
			for _, v := range scores {
				if math.IsNaN(float64(v)) {
					sawNaN = true
				}
			}
			if sawNaN != c.allNaN {
				t.Errorf("Softmax(%v) = %v, want allNaN=%v", c.in, scores, c.allNaN)
			}
			if !c.allNaN {
				// The ordinary case has to stay a distribution: -Inf underflows to zero
				// weight and the finite value takes all of it.
				var sum float64
				for _, v := range scores {
					sum += float64(v)
				}
				if math.Abs(sum-1) > 1e-6 {
					t.Errorf("Softmax(%v) = %v, sum %v, want 1", c.in, scores, sum)
				}
			}
		})
	}
}

// TestSiLUSaturatesAtBothEnds pins the activation's limits.
//
// SiLU is x * sigmoid(x), which tends to x for large positive and to 0 for large negative. The
// implementation clamps the exponent so a very negative input cannot overflow to +Inf and
// produce NaN; the clamps are what this checks, at magnitudes a real weight can reach.
func TestSiLUSaturatesAtBothEnds(t *testing.T) {
	x := []float32{-1000, -88, 0, 88, 1000}
	SiLU(x)
	for i, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("SiLU produced %v at index %d (input %v)", v, i, []float32{-1000, -88, 0, 88, 1000}[i])
		}
	}
	// The extremes are the function's limits: 0 on the far left, the input itself on the right.
	// -0 and 0 compare equal, and the implementation can produce either -- that is a sign bit,
	// not a value difference, so the assertion is on the number rather than its representation.
	if x[0] != 0 {
		t.Errorf("SiLU(-1000) = %v, want 0", x[0])
	}
	if math.Abs(float64(x[4]-1000)) > 1e-3 {
		t.Errorf("SiLU(1000) = %v, want ~1000", x[4])
	}
	// At zero the function is exactly zero, since x multiplies the sigmoid.
	if x[2] != 0 {
		t.Errorf("SiLU(0) = %v, want 0", x[2])
	}
}

// TestAddBias covers the bias pass that a Qwen2 model needs on q, k and v.
//
// The function had no test of its own. It is reached only when a model carries attention biases --
// the Llama fixtures do not -- so the end-to-end bias tests in the llama package were the only
// coverage, and they check the answer rather than the argument handling.
//
// Three behaviours are easy to get wrong and cheap to pin: the addition applies bias[i] to every
// row (not once), an empty bias is a no-op so callers need not branch on a bias-free model, and a
// bias longer than the row width is allowed and uses its prefix.
func TestAddBias(t *testing.T) {
	t.Run("applies to every row", func(t *testing.T) {
		dst := []float32{1, 2, 3, 4, 5, 6}
		if err := AddBias(dst, []float32{10, 20, 30}, 2, 3); err != nil {
			t.Fatalf("AddBias: %v", err)
		}
		want := []float32{11, 22, 33, 14, 25, 36}
		for i := range want {
			if dst[i] != want[i] {
				t.Errorf("dst[%d] = %v, want %v", i, dst[i], want[i])
			}
		}
	})

	t.Run("empty bias is a no-op", func(t *testing.T) {
		dst := []float32{1, 2, 3}
		if err := AddBias(dst, nil, 1, 3); err != nil {
			t.Fatalf("AddBias with no bias: %v", err)
		}
		for i, v := range []float32{1, 2, 3} {
			if dst[i] != v {
				t.Errorf("dst[%d] = %v, want %v (unchanged)", i, dst[i], v)
			}
		}
	})

	t.Run("longer bias uses its prefix", func(t *testing.T) {
		dst := []float32{1, 1}
		if err := AddBias(dst, []float32{5, 5, 5}, 1, 2); err != nil {
			t.Fatalf("AddBias with a long bias: %v", err)
		}
		if dst[0] != 6 || dst[1] != 6 {
			t.Errorf("dst = %v, want [6 6]", dst)
		}
	})

	t.Run("zero rows is a no-op", func(t *testing.T) {
		dst := []float32{1, 2}
		if err := AddBias(dst, []float32{9, 9}, 0, 2); err != nil {
			t.Fatalf("AddBias with no rows: %v", err)
		}
		if dst[0] != 1 || dst[1] != 2 {
			t.Errorf("dst = %v, want [1 2] (unchanged)", dst)
		}
	})
}

// TestAddBiasRejectsBadShapes covers the argument checks.
//
// rows is the only argument no slice length bounds -- width is limited by the bias and the
// destination, but a caller can pass any row count at all. That asymmetry made rows*width the one
// product here that can overflow, and it wrapped to a small positive number rather than a negative
// one, so the length check passed and the loop indexed past the destination:
//
//	AddBias(dst[:4096], bias[:4096], 2^53+1, 4096)  ->  slice bounds out of range [:8192]
//
// The check divides instead of multiplying now, which keeps both sides in range.
func TestAddBiasRejectsBadShapes(t *testing.T) {
	cases := []struct {
		name        string
		dst, bias   []float32
		rows, width int
	}{
		{"negative rows", make([]float32, 4), []float32{1, 2}, -1, 2},
		{"negative width", make([]float32, 4), []float32{1, 2}, 1, -2},
		{"bias shorter than width", make([]float32, 4), []float32{1}, 2, 2},
		{"dst too small", make([]float32, 3), []float32{1, 2}, 2, 2},
		// The overflow case: 2^53+1 rows of 4096 is far more than the destination holds, and the
		// product wraps to exactly the destination's length.
		{"row count whose product overflows", make([]float32, 4096), make([]float32, 4096), 1<<53 + 1, 4096},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("AddBias panicked instead of returning an error: %v", r)
				}
			}()
			if err := AddBias(c.dst, c.bias, c.rows, c.width); err == nil {
				t.Fatal("AddBias accepted invalid arguments")
			}
		})
	}
}

// TestSiLUBeyondTheClampIsZeroEitherWay pins what the clamp is for.
//
// The function clamps below -88 so exp never has to be evaluated on a value that would overflow.
// It is not a correctness guard: x/(1+Inf) is -0, not NaN, so the clamped and unclamped spellings
// agree on every input this can be handed. Measured at -89, -1000 and -1e30 -- all -0 without the
// clamp, all 0 with it.
//
// The test therefore asserts the value rather than the branch, and records that it cannot tell the
// two implementations apart. Moving the threshold from -88 to -8888 leaves this passing, and the
// comment on the function says why that is acceptable: the clamp earns its place on cost.
func TestSiLUBeyondTheClampIsZeroEitherWay(t *testing.T) {
	for _, v := range []float32{-89, -100, -1000, -1e30} {
		x := []float32{v}
		SiLU(x)
		if x[0] != 0 {
			t.Errorf("SiLU(%v) = %v, want 0", v, x[0])
		}
	}

	// And inside the clamp the function still computes, so a guard that blanked everything
	// negative cannot pass.
	for _, v := range []float32{-1, -0.5, 0, 0.5, 1} {
		x := []float32{v}
		SiLU(x)
		want := v / (1 + float32(math.Exp(float64(-v))))
		if x[0] != want {
			t.Errorf("SiLU(%v) = %v, want %v", v, x[0], want)
		}
	}

}

// TestKernelShapeErrorsStateTheNumbers covers the lengths in the shape-refusal messages.
//
// Each message already named the buffer, and the existing tests assert that much -- which of four
// buffers is short is the first question. The second is by how much, and that is what the numbers
// say: "dst has 1 values, need 2" is a one-element fix, while "dst has the wrong length" sends the
// reader back to the call site to count.
//
// Three functions, chosen because their messages print different shapes: a single pair, two resolved
// lengths and a row count, and the three-buffer MatMulTransposed.
func TestKernelShapeErrorsStateTheNumbers(t *testing.T) {
	const n, k = 2, 3

	t.Run("MatVec", func(t *testing.T) {
		err := MatVec(make([]float32, 1), make([]float32, n*k), make([]float32, k), n, k)
		if err == nil {
			t.Fatal("MatVec accepted a short destination")
		}
		// Both the length found and the length needed.
		for _, want := range []string{"has 1 values", "need 2"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not state %q", err, want)
			}
		}
	})

	t.Run("MatMulTransposed", func(t *testing.T) {
		const m, nn, kk = 2, 3, 2
		err := MatMulTransposed(make([]float32, m*nn-1), make([]float32, m*kk), make([]float32, nn*kk), m, nn, kk)
		if err == nil {
			t.Fatal("MatMulTransposed accepted a short destination")
		}
		// The length found and the length needed. MatMulTransposed prints the product rather than
		// its factors, unlike MatVec's rows-times-width form.
		for _, want := range []string{"has 5 values", "need 6"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not state %q", err, want)
			}
		}
	})

	t.Run("Dot", func(t *testing.T) {
		if _, err := Dot(make([]float32, 3), make([]float32, 5)); err == nil {
			t.Fatal("Dot accepted vectors of different lengths")
		} else if !strings.Contains(err.Error(), "3 and 5") {
			t.Errorf("error %q does not state both lengths", err)
		}
	})

	t.Run("AddBias", func(t *testing.T) {
		err := AddBias(make([]float32, 6), make([]float32, 2), 2, 3)
		if err == nil {
			t.Fatal("AddBias accepted a short bias")
		}
		if !strings.Contains(err.Error(), "has 2 values, need 3") {
			t.Errorf("error %q does not state both lengths", err)
		}

		// The destination check prints the row count and the row width, because the size it needs
		// is their product. Rows is the argument no slice bounds, which is why this check divides
		// rather than multiplies.
		err = AddBias(make([]float32, 5), make([]float32, 3), 2, 3)
		if err == nil {
			t.Fatal("AddBias accepted a short destination")
		}
		if !strings.Contains(err.Error(), "has 5 values, need 2 rows of 3") {
			t.Errorf("error %q does not state the size it needs", err)
		}
	})

	t.Run("RMSNorm", func(t *testing.T) {
		err := RMSNorm(make([]float32, 3), make([]float32, 4), make([]float32, 4), 1e-6)
		if err == nil {
			t.Fatal("RMSNorm accepted a short destination")
		}
		if !strings.Contains(err.Error(), "has 3 values, need 4") {
			t.Errorf("error %q does not state both lengths", err)
		}
	})
}

// TestSoftmaxAllInfiniteRow covers the one non-finite shape the test above leaves out: a row
// where every value is the same infinity.
//
// TestSoftmaxPropagatesNonFinite has "-Inf is ordinary" for a row with finite values beside it,
// and "+Inf as the maximum" for a row where one value is infinite. Neither puts an infinity on
// every element, which takes a different path through the reduction: max is then that infinity
// too, so the first subtraction is -Inf - -Inf (or +Inf - +Inf), and both are NaN. The row comes
// back all NaN, which is the same answer the NaN case gives and for the same reason -- a
// normaliser that cannot be computed leaves no distribution.
//
// The case is reachable from a checkpoint whose weights overflowed a conversion, and it is worth
// distinguishing from the mixed row because the mixed one is the one that WORKS: a finite value
// beside -Inf takes all the mass while -Inf contributes exp(-Inf) = 0. A test that only had the
// mixed row would suggest -Inf is always harmless.
func TestSoftmaxAllInfiniteRow(t *testing.T) {
	negInf := float32(math.Inf(-1))
	posInf := float32(math.Inf(1))

	for _, c := range []struct {
		name string
		in   []float32
	}{
		{"every value -Inf", []float32{negInf, negInf, negInf}},
		{"every value +Inf", []float32{posInf, posInf, posInf}},
	} {
		t.Run(c.name, func(t *testing.T) {
			scores := append([]float32(nil), c.in...)
			Softmax(scores)
			for i, v := range scores {
				if !math.IsNaN(float64(v)) {
					t.Errorf("Softmax(%v)[%d] = %v, want NaN: the maximum equals the value, so the "+
						"shift is an infinity minus itself", c.name, i, v)
				}
			}
		})
	}

	// The contrast, asserted in the same test so the two cannot drift apart: a single -Inf in a
	// row of finite values leaves a usable distribution, with the -Inf element at zero.
	mixed := []float32{0, negInf, 1}
	Softmax(mixed)
	if mixed[1] != 0 {
		t.Errorf("the -Inf element is %v, want 0", mixed[1])
	}
	var sum float64
	for _, v := range mixed {
		sum += float64(v)
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Errorf("the mixed row sums to %v, want 1", sum)
	}
	// And it must put the mass where the finite logits say: 1 has a higher logit than 0, so its
	// weight is larger. Without this the sum check would pass on a uniform row.
	if !(mixed[2] > mixed[0]) {
		t.Errorf("weights are %v and %v; the larger logit should take the larger share", mixed[2], mixed[0])
	}
}

// TestRMSNormAccumulatesInFloat64 covers the reason the sum of squares is not a float32.
//
// The accumulation is in float64 with a comment saying a 4096-wide float32 sum loses enough
// precision to change the normalized values, but nothing asserted it. The precision loss is
// the mild half. The severe half is range: a row of large activations squares to more than
// float32 can hold, so a float32 sum overflows to +Inf and every output of that row becomes
// zero -- a silently dead head, on a model whose weights are merely at the top of the range
// rather than corrupt.
//
// Measured: [3.4e38, 3.4e38] normalizes to [1.0000001, 1.0000001] in float64 accumulation,
// and to zero in a float32 one. The subnormal case is the other end, where squaring makes a
// value vanish entirely.
func TestRMSNormAccumulatesInFloat64(t *testing.T) {
	t.Run("large row does not overflow", func(t *testing.T) {
		// Two values at the top of float32's range: the sum of their squares is about
		// 2.3e77, far past float32's 3.4e38 limit but inside float64's.
		src := []float32{3.4e38, 3.4e38}
		dst := make([]float32, len(src))
		if err := RMSNorm(dst, src, []float32{1, 1}, 1e-5); err != nil {
			t.Fatalf("RMSNorm: %v", err)
		}
		for i, v := range dst {
			// Every element is scaled by the same factor, so all outputs are 1/sqrt(1 +
			// eps/mean) in magnitude -- about 1, and definitely not zero.
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("dst[%d] = %v, want a finite value", i, v)
				continue
			}
			if v < 0.9 || v > 1.1 {
				t.Errorf("dst[%d] = %v, want about 1: a float32 accumulation overflows here "+
					"and makes the row all zeros", i, v)
			}
		}
	})

	// A prediction about the low end was written and removed rather than kept: it assumed a
	// float32 sum would underflow a subnormal row to zero and produce a different scale, and
	// measurement showed the opposite -- 1e-38 squared is about 1e-76, which float64 holds
	// comfortably, so the mean is simply too small to matter against eps and the row normalizes
	// by 1/sqrt(eps) either way. A test asserting the guess would have failed on correct code.

}

// TestRMSNormZeroEpsOnAZeroRow pins the precondition its doc states.
//
// The doc used to say "eps prevents a division by zero on an all-zero row" without saying
// that it takes a positive eps to do it. The scale is 1/sqrt(mean+eps): with eps = 0 an
// all-zero row gives 1/0 = +Inf, and every weight becomes 0 * Inf = NaN. Measured, the row
// comes back as four NaNs where eps 1e-5 gives four zeros.
//
// RMSNorm does not refuse eps = 0 -- the caller owns that, and LoadConfig refuses a
// non-positive epsilon for every model this repository loads -- so the behaviour is pinned
// rather than guarded. Locking both halves means a future change that started clamping eps
// internally would show up here as a deliberate decision rather than a silent divergence
// from the doc.
func TestRMSNormZeroEpsOnAZeroRow(t *testing.T) {
	weight := []float32{1, 1, 1, 1}
	zero := []float32{0, 0, 0, 0}

	t.Run("a positive eps gives zeros", func(t *testing.T) {
		dst := make([]float32, 4)
		if err := RMSNorm(dst, zero, weight, 1e-5); err != nil {
			t.Fatalf("RMSNorm: %v", err)
		}
		for i, v := range dst {
			if v != 0 {
				t.Errorf("dst[%d] = %v, want 0: a zero row normalized by a positive eps is zero", i, v)
			}
		}
	})

	t.Run("a zero eps gives NaN", func(t *testing.T) {
		dst := make([]float32, 4)
		if err := RMSNorm(dst, zero, weight, 0); err != nil {
			t.Fatalf("RMSNorm: %v", err)
		}
		for i, v := range dst {
			if !math.IsNaN(float64(v)) {
				t.Errorf("dst[%d] = %v, want NaN: 0 * Inf is not zero", i, v)
			}
		}
	})
}

// TestMatVecAndOneRowMatMulAgree pins the equivalence the two forward paths rely on.
//
// llama has two forward paths and they reach different kernels for the same projection:
// forward.go calls MatVec for a single sequence (attention.go's cached path too), while
// batch.go calls MatMulTransposed for every batched prefill and decode. At the vocabulary
// projection that means the same weights are read through matVecRows on one path and through
// matMulTransposedCols with m=1 on the other. Nothing compared them.
//
// The gap matters because a divergence here is invisible end to end. Both kernels are
// parallel and both are bit-identical to their own serial form, so a test that compares
// batched output against batched output -- or single against single -- cannot see it. It
// would show up as a prompt whose logits change when a second request is admitted, which is
// the one thing TestInterleavingMatchesSequentialResults exists to rule out, and that test
// only exercises the batched path on both sides.
//
// The two implementations are not sharing code: one loops over rows of b with a scalar sum,
// the other tiles rows of a and indexes dst[i*n+j]. Both accumulate in float32 in the same
// order over the same products, which is what makes this an equality rather than an
// approximation.
func TestMatVecAndOneRowMatMulAgree(t *testing.T) {
	r := NewRunner(4)
	// Sizes that straddle the parallel threshold and the 8-row tile: k below it, k above it,
	// and an odd width that leaves a partial tile.
	shapes := []struct{ n, k int }{
		{n: 1, k: 1},
		{n: 3, k: 5},
		{n: 7, k: 64},
		{n: 130, k: 257},
		{n: 1000, k: 288},
	}
	for _, s := range shapes {
		t.Run(sizeName(s.n, s.k, "shape"), func(t *testing.T) {
			// The values have to be ones float32 cannot hold exactly, or the test proves
			// nothing: my first version used i%9/8 and i%5/4, every one a binary fraction,
			// and reversing the accumulation order left the comparison green. Irrational
			// and repeating fractions make the sum depend on the order, which is the
			// property under test -- the two kernels must accumulate in the same order, not
			// merely agree in exact arithmetic.
			a := make([]float32, s.n*s.k)
			for i := range a {
				a[i] = float32(math.Sin(float64(i)+1)) / 3
			}
			v := make([]float32, s.k)
			for i := range v {
				v[i] = float32(math.Cos(float64(i)+1)) * 0.7
			}
			want := make([]float32, s.n)
			got := make([]float32, s.n)
			if err := r.MatVec(want, a, v, s.n, s.k); err != nil {
				t.Fatalf("MatVec: %v", err)
			}
			if err := r.MatMulTransposed(got, v, a, 1, s.n, s.k); err != nil {
				t.Fatalf("MatMulTransposed: %v", err)
			}
			for j := range want {
				if got[j] != want[j] {
					t.Fatalf("column %d: one-row MatMulTransposed = %v, MatVec = %v",
						j, got[j], want[j])
				}
			}
		})
	}
}

// TestGELUValues checks the approximation against values worked out independently.
//
// The exact answers at zero and at the tails are what make this more than a
// self-consistency check: gelu(0) is 0 for any implementation, gelu(x) tends to x for large
// positive x and to 0 for large negative x, and a formula that dropped the cubic term or
// used the wrong constant still satisfies all three only by accident.
func TestGELUValues(t *testing.T) {
	near := func(got, want float32) bool {
		return math.Abs(float64(got-want)) < 1e-6
	}
	// gelu(0) = 0.
	if got := GELUValues(0); !near(got, 0) {
		t.Errorf("gelu(0) = %v, want 0", got)
	}
	// Far out, the two tails are exact: the tanh saturates.
	if got := GELUValues(20); !near(got, 20) {
		t.Errorf("gelu(20) = %v, want 20", got)
	}
	if got := GELUValues(-20); !near(got, 0) {
		t.Errorf("gelu(-20) = %v, want 0", got)
	}
	// At 1 the approximation and the exact definition are distinguishable, which is the point
	// of asserting both: the tanh form gives 0.84119, and 0.5*(1+erf(1/sqrt(2))) is 0.84134.
	//
	// Asserting the approximation's own value is deliberate. Asserting the exact one would
	// require erff and make this a different function; a tolerance wide enough to cover both
	// would not be able to tell the approximation from a formula with the cubic coefficient
	// dropped, which gives 0.831426 -- an order of magnitude further out than the gap between
	// the approximation and the exact function.
	const approx1 = 0.84119
	if got := GELUValues(1); math.Abs(float64(got-approx1)) > 1e-5 {
		t.Errorf("gelu(1) = %v, want the tanh approximation's %v", got, approx1)
	}
	// The minimum is about -0.170 at x = -0.752. A formula that clamped its output at zero,
	// which is a tempting simplification since GELU is nearly linear in both tails, would
	// answer 0 there and reduce the function's range by a fifth.
	if got := GELUValues(-0.751); got >= 0 {
		t.Errorf("gelu(-0.751) = %v, want the small negative minimum the real function has", got)
	}
}

// GELUValues returns the activation at one input, for the assertions above.
func GELUValues(x float32) float32 {
	v := []float32{x}
	GELU(v)
	return v[0]
}

// TestSoftcapValue checks the bound and the shape of the curve.
//
// The three properties that matter are separate: the value is bounded by c, it is unchanged near
// zero where the model's ordinary scores live, and it is odd (softcap(-v) == -softcap(v)), which
// is what keeps a bound from favouring one direction. A formula that forgot the factor outside
// tanh still satisfies the bound and the sign, so the exact values are asserted too.
func TestSoftcapValue(t *testing.T) {
	const c = 30
	near := func(got, want float32) bool { return math.Abs(float64(got-want)) < 1e-5 }

	// Odd, and zero at zero.
	if got := SoftcapValue(0, c); got != 0 {
		t.Errorf("softcap(0) = %v, want 0", got)
	}
	for _, v := range []float32{1, 12, 100, 1e6} {
		if pos, neg := SoftcapValue(v, c), SoftcapValue(-v, c); !near(pos, -neg) {
			t.Errorf("softcap is not odd at %v: %v and %v", v, pos, neg)
		}
	}
	// The bound is |result| <= c, and in float32 tanh reaches 1 on its own: 30*tanh(1e6/30) is
	// exactly 30, because 1e6/30 is far past the point where the float32 answer is 1. So the
	// equality case is reachable with a finite input, and the assertion has to allow it.
	//
	// Strictness is asserted where it is real: at 3c the curve is still inside the bound, which
	// is the range a trained model actually produces.
	cases := []struct {
		v      float32
		strict bool
	}{
		{100, true}, {1e6, false}, {float32(math.Inf(1)), false},
	}
	for _, tc := range cases {
		got := SoftcapValue(tc.v, c)
		if got > c || got < 0 {
			t.Errorf("softcap(%v) = %v, want inside [0, %v]", tc.v, got, c)
		}
		if tc.strict && got >= c {
			t.Errorf("softcap(%v) = %v, want strictly inside [0, %v)", tc.v, got, c)
		}
	}
	// Near zero it is the identity to first order: tanh(x) ~ x, so 1e-3 is unchanged.
	if got := SoftcapValue(1e-3, c); !near(got, 1e-3) {
		t.Errorf("softcap(1e-3) = %v, want it left alone", got)
	}
	// One exact point, computed independently: 30*tanh(31/30) for v = 31.
	if got, want := SoftcapValue(31, 30), float32(30*math.Tanh(31.0/30.0)); !near(got, want) {
		t.Errorf("softcap(31, 30) = %v, want %v", got, want)
	}

	// A non-positive cap is not a bound anyone could ask for: zero would divide by zero and a
	// negative one would invert the sign. Both leave the value alone.
	for _, bad := range []float32{0, -30} {
		if got := SoftcapValue(5, bad); got != 5 {
			t.Errorf("softcap(5, %v) = %v, want it left alone", bad, got)
		}
	}

	// The slice form agrees with the scalar one.
	vals := []float32{-100, -1, 0, 1, 100}
	want := make([]float32, len(vals))
	for i, v := range vals {
		want[i] = SoftcapValue(v, c)
	}
	Softcap(vals, c)
	for i := range vals {
		if vals[i] != want[i] {
			t.Errorf("Softcap[%d] = %v, want %v", i, vals[i], want[i])
		}
	}
}

// TestMatVecRejectsOverflowingShape covers the shape whose product wraps.
//
// n*k overflows int when both are large, so the guard `len(a) < n*k` would compare against a
// negative number and pass. Found while writing FuzzMatVecShapes: with n = MaxInt and k = 2,
// n*k is -2.
//
// What stops it is the order of the checks rather than a dedicated overflow guard: the dst check
// is `len(dst) < n`, and n is a single dimension, so a shape large enough to overflow also demands
// a dst that cannot exist. Measured, checkMatVec(nil, nil, make([]float32, 2), MaxInt, 2) fails on
// dst rather than on a. That is a real property and a fragile one -- moving the dst comparison
// below the a comparison would open the hole -- so it is asserted here rather than left implicit.
func TestMatVecRejectsOverflowingShape(t *testing.T) {
	n, k := math.MaxInt, 2
	if got := n * k; got >= 0 {
		t.Fatalf("n*k did not overflow (%d), so this test no longer tests what it says", got)
	}
	// Every buffer long enough to pass the checks that come before the overflowing one, which is
	// the combination that would reach a write if the ordering changed.
	err := MatVec(make([]float32, 2), make([]float32, 2), make([]float32, k), n, k)
	if err == nil {
		t.Fatal("MatVec accepted a shape whose product overflows int")
	}
	// The refusal names a dimension, so the message is usable rather than a bare failure. The
	// assertion used to check only the "kernel:" prefix while this comment claimed the dimension
	// was named.
	if !strings.Contains(err.Error(), "kernel:") {
		t.Errorf("error = %q, want it prefixed like the package's other shape errors", err)
	}
	if !strings.Contains(err.Error(), strconv.Itoa(n)) {
		t.Errorf("error = %q, want it to name the %d it refused", err, n)
	}
	if !strings.Contains(err.Error(), "dst") {
		t.Errorf("error = %q, want it to name which buffer was short", err)
	}
}
