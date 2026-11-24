package kernel

import (
	"fmt"
	"math"
	"slices"
)

// MatMul computes dst = a @ b for row-major matrices.
//
// Shapes: a is m x k, b is k x n, dst is m x n. This is the reference
// implementation: three nested loops, no blocking, no unrolling. Every
// optimized version added later must produce the same result and is checked
// against this one.
//
// The accumulation order is i, p, j — the natural reading of "for each output
// element, walk the shared dimension". Floating-point addition is not
// associative, so this order is part of the function's contract: a reordered
// implementation is a different function numerically, not just a different
// implementation. Tests compare against this one with a tolerance that
// accounts for that.
func MatMul(dst, a, b []float32, m, n, k int) error {
	if err := checkMatMul(dst, a, b, m, n, k); err != nil {
		return err
	}
	for i := range m {
		dstRow := dst[i*n : (i+1)*n]
		// Zero the row first: MatMul assigns, it does not accumulate.
		for j := range dstRow {
			dstRow[j] = 0
		}
		aRow := a[i*k : (i+1)*k]
		for p := range k {
			av := aRow[p]
			if av == 0 {
				continue
			}
			bRow := b[p*n : (p+1)*n]
			for j := range n {
				dstRow[j] += av * bRow[j]
			}
		}
	}
	return nil
}

// MatMulTransposed computes dst = a @ bᵀ where b is stored as n x k.
//
// A transformer's linear layers usually need the weight matrix in its
// transposed orientation: ggml stores a projection as [in, out] and the
// forward pass multiplies an activation row by it. Providing the transposed
// form directly avoids materializing a k x n copy, which for a 7B model's
// FFN would be most of the layer's memory.
func MatMulTransposed(dst, a, b []float32, m, n, k int) error {
	if err := checkMatMulTransposed(dst, a, b, m, n, k); err != nil {
		return err
	}
	matMulTransposedCols(dst, a, b, m, n, k, 0, n)
	return nil
}

// matMulTransposedCols computes columns [jlo,jhi) of MatMulTransposed, all m rows.
//
// The parallel split is over columns of the output, not over its rows. A row split
// cannot use more than m workers, and the decode case that matters most has m = 1:
// one sequence, one activation row, the whole vocabulary as n. Splitting by row there
// leaves the entire product on a single goroutine, which made single-stream decode
// about 2.2x slower than a per-element split once the tiling landed. Every output
// element depends only on its own row of a and its own column of b, so either
// direction splits without changing a value; columns is the one that always has work
// to hand out, because n is the vocabulary size in the projection that dominates a
// decode step.
//
// Inside a column range the rows are still visited in tiles, so a row of b is loaded
// once and used for every row of a in the tile rather than re-read per output row.
// That is the memory-traffic win the tiling exists for, and it survives: each worker
// owns a contiguous band of columns and reuses b within it.
//
// Neither the tiling nor the split changes the order of the p loop, so the result stays
// bit-identical to a naive element-at-a-time multiply.
func matMulTransposedCols(dst, a, b []float32, m, n, k, jlo, jhi int) {
	const rowTile = 8
	for base := 0; base < m; base += rowTile {
		end := min(base+rowTile, m)
		for j := jlo; j < jhi; j++ {
			bRow := b[j*k : (j+1)*k]
			for i := base; i < end; i++ {
				aRow := a[i*k : (i+1)*k]
				var sum float32
				for p := range k {
					sum += aRow[p] * bRow[p]
				}
				dst[i*n+j] = sum
			}
		}
	}
}

// MatVec computes dst = a @ v for a weight matrix stored as n x k.
//
// This is the case that dominates token generation: one activation vector
// against every projection in the model, once per token. It is a separate
// function rather than a 1-row MatMul because it is the hot path and because
// its shape (no m dimension) is easier to reason about.
//
// This is the serial reference. Runner.MatVec computes the same values across
// several goroutines and is bit-identical, because the rows are independent.
func MatVec(dst, a, v []float32, n, k int) error {
	if err := checkMatVec(dst, a, v, n, k); err != nil {
		return err
	}
	matVecRows(dst, a, v, n, k, 0, n)
	return nil
}

// matVecRows computes rows [lo,hi) of MatVec. Each row is a complete dot
// product in serial order, so a caller may split the row range across
// goroutines without changing any value.
func matVecRows(dst, a, v []float32, n, k, lo, hi int) {
	for j := lo; j < hi; j++ {
		row := a[j*k : (j+1)*k]
		var sum float32
		for p := range k {
			sum += row[p] * v[p]
		}
		dst[j] = sum
	}
}

// AddBias adds a bias vector to every row of dst, in place.
//
// A projection with a bias is dst = a @ bᵀ + bias, and adding it as a separate pass
// keeps the matmul kernel free of a branch every model would pay for even when it
// has no bias. rows may be 1 (decode) or the batch size (prefill).
//
// An empty bias is a no-op, so a caller with a model that has no bias does not need
// to check before calling.
func AddBias(dst, bias []float32, rows, width int) error {
	if rows < 0 || width < 0 {
		return fmt.Errorf("kernel: negative shape in AddBias(%d, %d)", rows, width)
	}
	if len(bias) == 0 {
		return nil
	}
	if len(bias) < width {
		return fmt.Errorf("kernel: bias has %d values, need %d", len(bias), width)
	}
	// rows is the only argument here that no slice length bounds, so rows*width can overflow int --
	// and it does so to a small positive value, not a negative one, which is why the comparison below
	// used to pass and the loop then indexed past dst. Dividing instead of multiplying keeps both
	// sides of the comparison in range.
	if width > 0 && rows > len(dst)/width {
		return fmt.Errorf("kernel: dst has %d values, need %d rows of %d", len(dst), rows, width)
	}
	for r := range rows {
		row := dst[r*width : (r+1)*width]
		for i := range width {
			row[i] += bias[i]
		}
	}
	return nil
}

// RMSNorm normalizes src in place into dst using the root-mean-square norm.
//
// dst and src may alias. Each element is divided by the RMS of the row and
// then multiplied by the corresponding weight, which is the normalization
// used by Llama-family models.
//
// eps prevents the division by zero on an all-zero row, and it needs to be positive to do
// that: the scale is 1/sqrt(mean+eps), so eps = 0 gives 1/0 = +Inf and every weight becomes
// 0 * Inf = NaN. Measured on a zero row, eps 0 returns NaN and eps 1e-5 returns zeros. This
// function does not refuse a zero eps -- every caller passes cfg.RMSNormEps, which LoadConfig
// refuses at or below zero -- so the requirement is stated here rather than enforced twice.
func RMSNorm(dst, src, weight []float32, eps float32) error {
	n := len(src)
	if len(dst) < n {
		return fmt.Errorf("kernel: dst has %d values, need %d", len(dst), n)
	}
	if len(weight) < n {
		return fmt.Errorf("kernel: weight has %d values, need %d", len(weight), n)
	}
	if n == 0 {
		return nil
	}

	var sumSquares float64
	for _, v := range src {
		sumSquares += float64(v) * float64(v)
	}
	mean := sumSquares / float64(n)
	// Accumulate in float64: summing squares in float32 for a 4096-wide row
	// loses enough precision to change the normalized values.
	scale := float32(1 / math.Sqrt(mean+float64(eps)))

	for i := range n {
		dst[i] = src[i] * scale * weight[i]
	}
	return nil
}

// Softmax replaces scores with a probability distribution in place.
//
// It subtracts the maximum first so that exp cannot overflow: the largest
// score becomes exp(0) = 1 and everything else is smaller. Without that step
// a logit above ~88 makes exp return +Inf and the whole row becomes NaN.
func Softmax(scores []float32) {
	if len(scores) == 0 {
		return
	}
	// The max of the whole slice; slices.Max handles the comparison and reads
	// as the intent rather than as a loop that happens to compute it.
	max := slices.Max(scores)

	var sum float64
	for i, v := range scores {
		e := math.Exp(float64(v - max))
		scores[i] = float32(e)
		sum += e
	}
	if sum == 0 {
		// Not reachable from any input this package has been given: the maximum element
		// contributes exp(0) = 1, so the sum is at least 1 unless max is NaN or +Inf, and
		// then the sum is NaN rather than zero. Kept as a guard because the invariant is
		// about float64 accumulation and not about the slice, which is the part a future
		// edit could change -- and dividing by zero here would turn one bad value into a
		// uniform distribution that looks like a real answer.
		//
		// An earlier version of this comment said it was "only reachable with NaN input",
		// which is backwards: NaN skips the guard rather than reaching it.
		return
	}
	inv := float32(1 / sum)
	for i := range scores {
		scores[i] *= inv
	}
}

// SiLU applies the sigmoid linear unit, x * sigmoid(x), in place.
//
// This is the activation in Llama's gated feed-forward block. Computed as x / (1 + exp(-x)),
// with inputs below -88 answered as zero without evaluating exp at all.
//
// The clamp is a speed choice, not a correctness one, and the comment here used to say
// otherwise -- that without it exp would overflow to +Inf and produce a NaN. Measured, it
// does not: the exponent is taken in float64 and only the result is narrowed, so even
// x = -3.4e38 gives -0 rather than NaN. What the clamp buys is the cost of that exp on a
// value far outside the sigmoid's useful range: over a million values of -200, the clamped
// form took 0.85ms against 35.7ms unclamped, 42x. It also costs a branch on ordinary inputs
// -- 30ms against 18ms over a million values in [-3, 3] -- which is why the threshold sits
// where the answer is already 5e-37 rather than nearer the interesting range.
func SiLU(x []float32) {
	for i, v := range x {
		// Below this the sigmoid is 1 to well past float32's precision, so the product is
		// zero for every purpose but exact reproduction of a denormal.
		if v < -88 {
			x[i] = 0
			continue
		}
		x[i] = v / (1 + float32(math.Exp(float64(-v))))
	}
}

// GELU applies the Gaussian error linear unit in place, using the tanh approximation
// 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))).
//
// This is Gemma's feed-forward gate. The approximation is the one the reference implementations
// use: the exact form needs erff, and the two differ by 1.5e-4 at 1 and less elsewhere over the
// range a gate sees, far below what inference is sensitive to.
//
// Unlike SiLU there is no clamp. tanh saturates on its own, so a value far from zero costs the
// same as any other and neither tail is truncated.
func GELU(x []float32) {
	// sqrt(2/pi), the constant the approximation is defined with.
	const scale = 0.7978845608028654
	// The cubic coefficient from the approximation.
	const cubic = 0.044715
	for i, v := range x {
		f := float64(v)
		inner := scale * (f + cubic*f*f*f)
		x[i] = 0.5 * v * (1 + float32(math.Tanh(inner)))
	}
}

// SoftcapValue bounds one value with c*tanh(v/c), for c > 0.
//
// The curve is near-linear for |v| well inside c and saturates towards +/-c outside it, so a
// model that uses it keeps ordinary scores untouched while an extreme one cannot run away. A
// non-positive c returns v unchanged: zero would divide by zero and a negative cap would invert
// the sign, and neither is a bound anything would ask for.
func SoftcapValue(v, c float32) float32 {
	if c <= 0 {
		return v
	}
	return c * float32(math.Tanh(float64(v/c)))
}

// Softcap applies SoftcapValue to every element in place.
func Softcap(x []float32, c float32) {
	if c <= 0 {
		return
	}
	for i, v := range x {
		x[i] = SoftcapValue(v, c)
	}
}

// AddInPlace adds b to a elementwise, a += b. Both must have the same length.
func AddInPlace(a, b []float32) error {
	if len(a) != len(b) {
		return fmt.Errorf("kernel: AddInPlace lengths differ: %d and %d", len(a), len(b))
	}
	for i := range a {
		a[i] += b[i]
	}
	return nil
}

// MulInPlace multiplies a by b elementwise, a *= b.
func MulInPlace(a, b []float32) error {
	if len(a) != len(b) {
		return fmt.Errorf("kernel: MulInPlace lengths differ: %d and %d", len(a), len(b))
	}
	for i := range a {
		a[i] *= b[i]
	}
	return nil
}

// Dot returns the inner product of a and b.
func Dot(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("kernel: Dot lengths differ: %d and %d", len(a), len(b))
	}
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum, nil
}

// checkMatMul validates the buffer sizes for an m x n x k product.
func checkMatMul(dst, a, b []float32, m, n, k int) error {
	if m < 0 || n < 0 || k < 0 {
		return fmt.Errorf("kernel: negative dimension in %dx%dx%d", m, n, k)
	}
	if len(dst) < m*n {
		return fmt.Errorf("kernel: dst has %d values, need %d", len(dst), m*n)
	}
	if len(a) < m*k {
		return fmt.Errorf("kernel: a has %d values, need %d", len(a), m*k)
	}
	if b != nil && len(b) < k*n {
		return fmt.Errorf("kernel: b has %d values, need %d", len(b), k*n)
	}
	return nil
}
