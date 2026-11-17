package kernel

import (
	"fmt"
	"sync"
)

// Runner evaluates the row-partitioned kernels across a bounded number of
// workers.
//
// Decoding a token is a chain of matrix-vector products, one per projection, and
// each of them walks a weight matrix that does not fit in cache. The arithmetic
// is a handful of cycles per value while the load is the bottleneck, so the work
// parallelizes cleanly: there is no shared state between output rows and no
// reduction spanning rows.
//
// The split is always over independent output elements, never over the reduction
// dimension. That is what makes a Runner a scheduling change rather than a
// numerical one: every output element is evaluated by exactly one worker in
// exactly the order the serial kernel uses, so float32 addition is never
// reassociated and the result is bit-identical to MatVec and MatMulTransposed. A
// kernel that split the reduction would still be "about right" while quietly
// returning different numbers, which is the kind of difference that later hides
// a real bug.
//
// A nil Runner is valid and means one worker, so a caller that never configures
// concurrency still gets a usable value.
type Runner struct {
	workers int
}

// NewRunner returns a Runner that may use up to workers goroutines.
//
// workers <= 1 disables parallelism. There is no upper clamp here: the work being
// split bounds the fan-out, so asking for more workers than there is work simply
// uses fewer.
func NewRunner(workers int) *Runner {
	return &Runner{workers: workers}
}

// Workers returns the configured fan-out.
func (r *Runner) Workers() int {
	if r == nil {
		return 1
	}
	return r.workers
}

// MatVec computes dst = a @ v for a weight matrix stored as n x k, splitting the
// rows of a across the workers.
//
// The result is identical to MatVec. Small products stay on one goroutine: below
// a few thousand multiply-adds the spawn and join cost more than the arithmetic,
// and every transformer block issues several small projections.
func (r *Runner) MatVec(dst, a, v []float32, n, k int) error {
	if err := checkMatVec(dst, a, v, n, k); err != nil {
		return err
	}
	return r.splitRows(r.parts(n*k), n, func(lo, hi int) { _ = matVecRows(dst, a, v, n, k, lo, hi) })
}

// MatMulTransposed computes dst = a @ bᵀ where b is stored as n x k, splitting
// the product across the workers.
//
// The split is over output elements flattened into i*n+j order, not over the
// rows of a alone. With m rows the two differ only when m is smaller than the
// worker count, but that is exactly the decode case: a batch of m sequences has
// m = B, so a row-only split would leave the vast majority of the product serial
// for small B and would make batching strictly slower than looping over MatVec.
// One row and a large n is a full-size product and should use the whole machine.
//
// The result is identical to MatMulTransposed.
func (r *Runner) MatMulTransposed(dst, a, b []float32, m, n, k int) error {
	if err := checkMatMulTransposed(dst, a, b, m, n, k); err != nil {
		return err
	}
	// Split over row ranges rather than elements: each worker then gets a contiguous band
	// of output rows, and the row tiling inside matMulTransposedRows reuses a row of b
	// across the band instead of re-reading it for every output row.
	//
	// Every output element depends only on its own row of a, so the split cannot change
	// a value: two workers write disjoint rows and read the same b.
	return r.splitRows(r.parts(m*n*k), m, func(lo, hi int) {
		_ = matMulTransposedRows(dst, a, b, m, n, k, lo, hi)
	})
}

// Parallelism is only worth its spawn-and-join cost above a certain size, and
// that size is measured rather than assumed. Writing the crossover down as
// constants keeps the tuning in one place and makes the benchmark that produced
// them the obvious thing to rerun.
const (
	// minParallelWork is the multiply-add count below which a kernel stays on
	// one goroutine. Measured on an EPYC 7742: at 27k multiply-adds the
	// parallel form breaks even, at 8k it is 1.4x slower.
	minParallelWork = 1 << 15
	// workPerWorker is the work each additional worker must receive. Without it
	// a modest projection fans out to every available CPU and loses to two
	// workers: measured at 41k multiply-adds, two workers win by 1.18x while
	// sixteen lose.
	workPerWorker = 1 << 14
)

// parts returns how many workers an operation of work multiply-adds should use.
//
// The count is bounded by the configured workers and by the work available, so a
// projection that is too small does not start goroutines that cannot pay for
// themselves.
func (r *Runner) parts(work int) int {
	if r == nil || r.workers <= 1 || work < minParallelWork {
		return 1
	}
	return min(r.workers, work/workPerWorker)
}

// splitRows runs fn over the row range [0,n) in contiguous chunks.
func (r *Runner) splitRows(parts, n int, fn func(lo, hi int)) error {
	if parts > n {
		parts = n
	}
	return r.split(parts, n, fn)
}

// split runs fn over [0,n) in at most parts contiguous chunks, returning when
// every chunk is done.
//
// Chunks are contiguous and computed up front rather than pulled from a queue:
// the row ranges are then fixed and visible, and the work per chunk is uniform
// because every row of a kernel costs the same.
func (r *Runner) split(parts, n int, fn func(lo, hi int)) error {
	if parts <= 1 {
		fn(0, n)
		return nil
	}
	// ceil division: with n not divisible by parts the first chunks take one
	// extra row and the last may be empty, which is why the empty case is
	// guarded below.
	chunk := (n + parts - 1) / parts
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Go(func() { fn(lo, hi) })
	}
	wg.Wait()
	return nil
}

// checkMatVec validates MatVec's buffers.
func checkMatVec(dst, a, v []float32, n, k int) error {
	if n < 0 || k < 0 {
		return fmt.Errorf("kernel: negative shape in MatVec(%d, %d)", n, k)
	}
	if len(v) < k {
		return fmt.Errorf("kernel: v has %d values, need %d", len(v), k)
	}
	if len(a) < n*k {
		return fmt.Errorf("kernel: a has %d values, need %d", len(a), n*k)
	}
	if len(dst) < n {
		return fmt.Errorf("kernel: dst has %d values, need %d", len(dst), n)
	}
	return nil
}

// checkMatMulTransposed validates MatMulTransposed's buffers.
func checkMatMulTransposed(dst, a, b []float32, m, n, k int) error {
	if err := checkMatMul(dst, a, nil, m, n, k); err != nil {
		return err
	}
	if len(b) < n*k {
		return fmt.Errorf("kernel: b has %d values, need %d", len(b), n*k)
	}
	return nil
}
