package kernel

import (
	"math"
	"math/rand"
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
// matMulTransposedRows depends on: the order output rows are visited changes, but each
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
