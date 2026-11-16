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
	return matMulTransposedRows(dst, a, b, m, n, k, 0, m)
}

// matMulTransposedRows computes rows [lo,hi) of MatMulTransposed. Every output
// element depends only on its own row of a, so a caller may split the row range
// without changing any value.
func matMulTransposedRows(dst, a, b []float32, m, n, k, lo, hi int) error {
	for i := lo; i < hi; i++ {
		aRow := a[i*k : (i+1)*k]
		dstRow := dst[i*n : (i+1)*n]
		for j := range n {
			bRow := b[j*k : (j+1)*k]
			var sum float32
			for p := range k {
				sum += aRow[p] * bRow[p]
			}
			dstRow[j] = sum
		}
	}
	return nil
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
	return matVecRows(dst, a, v, n, k, 0, n)
}

// matVecRows computes rows [lo,hi) of MatVec. Each row is a complete dot
// product in serial order, so a caller may split the row range across
// goroutines without changing any value.
func matVecRows(dst, a, v []float32, n, k, lo, hi int) error {
	for j := lo; j < hi; j++ {
		row := a[j*k : (j+1)*k]
		var sum float32
		for p := range k {
			sum += row[p] * v[p]
		}
		dst[j] = sum
	}
	return nil
}

// RMSNorm normalizes src in place into dst using the root-mean-square norm.
//
// dst and src may alias. Each element is divided by the RMS of the row and
// then multiplied by the corresponding weight, which is the normalization
// used by Llama-family models. eps prevents a division by zero on an
// all-zero row.
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
		// Only reachable with NaN input; leave the row alone rather than
		// dividing by zero and turning a bad value into a silent uniform
		// distribution.
		return
	}
	inv := float32(1 / sum)
	for i := range scores {
		scores[i] *= inv
	}
}

// SiLU applies the sigmoid linear unit, x * sigmoid(x), in place.
//
// This is the activation in Llama's gated feed-forward block. Computed as
// x / (1 + exp(-x)) with the argument clamped, so a large negative input
// cannot make exp overflow to +Inf and produce a NaN.
func SiLU(x []float32) {
	for i, v := range x {
		// exp overflows float32 above ~88; beyond that the function is
		// indistinguishable from zero anyway.
		if v < -88 {
			x[i] = 0
			continue
		}
		x[i] = v / (1 + float32(math.Exp(float64(-v))))
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
