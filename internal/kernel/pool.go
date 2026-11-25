package kernel

import (
	"fmt"
	"sync"
)

// Runner evaluates the partitionable kernels across a bounded number of workers.
//
// Decoding a token is a chain of matrix-vector products, one per projection, and
// each of them walks a weight matrix that does not fit in cache. The arithmetic
// is a handful of cycles per value while the load is the bottleneck, so the work
// parallelizes cleanly: there is no shared state between output elements and no
// reduction spanning them.
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
	return r.splitRows(r.parts(n*k), n, func(lo, hi int) { matVecRows(dst, a, v, n, k, lo, hi) })
}

// MatMulTransposed computes dst = a @ bᵀ where b is stored as n x k, splitting
// the product across the workers.
//
// The split is over output columns, not over the rows of a. With m rows a row split
// can use at most m workers, and the decode case that dominates is m = 1: a single
// sequence's activation row against the whole vocabulary. Splitting by row there
// leaves the product serial, which is 2.2x slower than it needs to be, while columns
// scale with n and n is the vocabulary. Every output element depends only on its own
// row of a and its own column of b, so neither direction can change a value.
//
// The result is identical to MatMulTransposed.
func (r *Runner) MatMulTransposed(dst, a, b []float32, m, n, k int) error {
	if err := checkMatMulTransposed(dst, a, b, m, n, k); err != nil {
		return err
	}
	// Split over column ranges: each worker gets a contiguous band of output columns and
	// the row tiling inside matMulTransposedCols reuses a row of b across the band.
	parts := r.parts(m * n * k)
	if splitOverRows(m, n, parts) {
		return r.splitRows(parts, m, func(lo, hi int) {
			for j := range n {
				for i := lo; i < hi; i++ {
					var sum float32
					aRow := a[i*k : (i+1)*k]
					bRow := b[j*k : (j+1)*k]
					for p := range k {
						sum += aRow[p] * bRow[p]
					}
					dst[i*n+j] = sum
				}
			}
		})
	}
	return r.splitCols(parts, n, func(jlo, jhi int) {
		matMulTransposedCols(dst, a, b, m, n, k, jlo, jhi)
	})
}

// splitOverRows reports whether MatMulTransposed should split its output by rows rather
// than by columns.
//
// Columns are the default because n is the vocabulary in the projection that dominates a
// decode step, so a column split always has work to hand out. Rows win only in the one
// shape where they give every worker more to do than columns would: a tall product whose
// columns are too few to divide among the workers, as when a batched prefill multiplies a
// long prompt by a short intermediate. Even then it is a preference, not a requirement --
// a row split can only ever use m workers, which is why it is never the fallback for the
// m = 1 case that made single-stream decode 2.2x slower.
//
// This function exists so that decision is a single named predicate a test can pin. The
// regression it guards against was not a wrong value but a wrong axis, and a wrong axis
// leaves every output bit-identical, so no comparison against the serial kernel could see
// it.
func splitOverRows(m, n, parts int) bool {
	if parts <= 1 {
		return false
	}
	if m <= 1 {
		// One output row: a row split would put the whole product on one worker.
		return false
	}
	// Rows win when they can occupy at least as many workers as columns can, and each
	// worker still gets a row. n >= m means columns are at least as parallel, so the
	// default stands.
	return m > n
}

// Parallelism is only worth its spawn-and-join cost above a certain size, and
// that size is measured rather than assumed. Writing the crossover down as
// constants keeps the tuning in one place and makes the benchmark that produced
// them the obvious thing to rerun.
const (
	// minParallelWork is the multiply-add count below which a kernel stays on
	// one goroutine. Measured on an EPYC 7742: the parallel form breaks even at
	// 27k multiply-adds and is 1.2x slower at 8k. The constant is the next power
	// of two above the break-even point, so the threshold errs toward serial --
	// a kernel one step below it pays the call overhead without the fan-out.
	minParallelWork = 1 << 15
	// workPerWorker is the work each additional worker must receive. Without it a modest
	// projection fans out to every available CPU and loses to two workers: measured at 41k
	// multiply-adds, two workers win by 1.18x while sixteen lose.
	//
	// With it, 41k gives parts() = 2 whatever the runner was configured for, so the sixteen-way
	// case cannot recur. BenchmarkWorkPerWorker measures that: every worker count from 2 upward
	// reports 2 parts, which is the guard working rather than a measurement of why sixteen would
	// have lost.
	//
	// Re-measuring it needs an idle machine. The host this was last checked on was saturated
	// (load average 131 on 128 cores), and under that load every parallel path in this package
	// measured slower than the serial one -- the comparison is then about waiting for a core
	// rather than about the fan-out. The number above is what an EPYC 7742 produced.
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

// Parallel runs fn over [0,count) across the runner's workers and returns when every call
// is done.
//
// fn receives the index and the slot of the worker running it: slots are distinct for
// concurrently running workers, so a caller that needs per-worker scratch indexes an
// array by slot. Passing the slot rather than letting the caller derive it is what makes
// that safe -- a caller guessing from the index cannot know how this function chunked the
// range, and a wrong guess is two workers writing the same scratch.
//
// It exists for work that is independent per index but is not a projection: attention over
// one position's heads, for example, where each position writes its own slice of the
// output and reads only the cache. The caller owns the disjointness of the output; this
// only owns the distinctness of the slots.
func (r *Runner) Parallel(count int, fn func(i, slot int)) {
	if count <= 0 {
		return
	}
	workers := 1
	if r != nil {
		workers = r.workers
	}
	parts := min(workers, count)
	if parts <= 1 {
		for i := range count {
			fn(i, 0)
		}
		return
	}
	// ceil division: with count not divisible by parts the first chunks take one extra
	// index, so the ranges stay contiguous and cover [0,count) exactly once.
	chunk := (count + parts - 1) / parts
	var wg sync.WaitGroup
	for n, lo := 0, 0; lo < count; lo, n = lo+chunk, n+1 {
		hi := min(lo+chunk, count)
		// slot is passed as a parameter rather than captured: a closure over a variable
		// the loop mutates makes every goroutine read whatever value it holds when it
		// runs, which is how two workers end up sharing one slot.
		wg.Go(func() {
			for i := lo; i < hi; i++ {
				fn(i, n)
			}
		})
	}
	wg.Wait()
}

// splitRows runs fn over the row range [0,n) in contiguous chunks.
func (r *Runner) splitRows(parts, n int, fn func(lo, hi int)) error {
	if parts > n {
		parts = n
	}
	return r.split(parts, n, fn)
}

// splitCols runs fn over the column range [0,n) in contiguous chunks.
//
// It is splitRows under another name, and the name is the point: MatMulTransposed chooses
// between them by axis, splitting rows when the row tiling wins and columns otherwise, and a
// reader who saw `splitRows` in that choice would have to work out why a row split is right
// there. The mechanics are identical, so there is nothing to keep in sync.
func (r *Runner) splitCols(parts, n int, fn func(lo, hi int)) error {
	return r.splitRows(parts, n, fn)
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
