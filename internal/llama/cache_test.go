package llama

import (
	"strings"
	"testing"
)

// TestNewKVCacheRejectsBadGeometry covers the guard on the config the cache sizes itself from.
//
// Three fields are checked and each is a separate way the allocation arithmetic goes wrong: no
// layers, no key/value heads, or a zero head width. Losing any one of them leaves the product at zero,
// so the cache would allocate nothing and every store would write past it.
//
// The message prints the whole config rather than the failing field, because a caller reading it has
// the struct and can see which is zero.
func TestNewKVCacheRejectsBadGeometry(t *testing.T) {
	good := Config{Layers: 2, KVHeads: 2, HeadDim: 8}

	if _, err := NewKVCache(good, 8); err != nil {
		t.Fatalf("NewKVCache refused a valid config: %v", err)
	}

	cases := []struct {
		name string
		cfg  Config
	}{
		{"no layers", Config{Layers: 0, KVHeads: 2, HeadDim: 8}},
		{"no kv heads", Config{Layers: 2, KVHeads: 0, HeadDim: 8}},
		{"no head width", Config{Layers: 2, KVHeads: 2, HeadDim: 0}},
		{"negative layers", Config{Layers: -1, KVHeads: 2, HeadDim: 8}},
		{"negative kv heads", Config{Layers: 2, KVHeads: -1, HeadDim: 8}},
		{"negative head width", Config{Layers: 2, KVHeads: 2, HeadDim: -8}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cache, err := NewKVCache(c.cfg, 8)
			if err == nil {
				t.Fatalf("NewKVCache accepted %+v", c.cfg)
			}
			if cache != nil {
				t.Error("a refused config returned a cache")
			}
			if !strings.Contains(err.Error(), "valid config") {
				t.Errorf("error %q does not say the config is the problem", err)
			}
		})
	}
}

// TestNewKVCacheRejectsANonPositiveSize is separate from the geometry cases because it is checked
// first, and its message names the size rather than the config.
func TestNewKVCacheRejectsANonPositiveSize(t *testing.T) {
	cfg := Config{Layers: 2, KVHeads: 2, HeadDim: 8}
	for _, n := range []int{0, -1} {
		cache, err := NewKVCache(cfg, n)
		if err == nil {
			t.Errorf("NewKVCache accepted nCtx=%d", n)
			continue
		}
		if cache != nil {
			t.Errorf("nCtx=%d returned a cache alongside the error", n)
		}
		if !strings.Contains(err.Error(), "cache size must be positive") {
			t.Errorf("nCtx=%d: error %q does not say the size is the problem", n, err)
		}
	}
}

// TestSpansOfZeroRowsIsEmptyNotAPanic pins the guard whose comment says why it exists.
//
// spans computes end as off + (n-1)*dim + head. With n = 0 that is off - dim + head, which
// is negative, and the slice expression c.k[layer][off:end] panics. The comment calls it a
// contract rather than a forward-pass path -- every caller in this package asks for at
// least one row -- but a public method that panics on a legal zero is still a crash, so the
// guard returns no spans instead.
//
// Verified by deleting the guard: the call panics with "slice bounds out of range [:-16]"
// for a 16-wide head. It had no test, so nothing would have caught a "simplification" that
// folded the zero case into the general path.
func TestSpansOfZeroRowsIsEmptyNotAPanic(t *testing.T) {
	// A constructed config rather than a fixture: spans only reads the geometry, and this
	// package's other cache tests build one the same way.
	cfg := Config{Layers: 1, KVHeads: 1, HeadDim: 16}
	cache, err := NewKVCache(cfg, 8)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}

	keys, values, err := cache.spans(0, 0, 0)
	if err != nil {
		t.Fatalf("spans(0 rows): %v", err)
	}
	if len(keys) != 0 || len(values) != 0 {
		t.Errorf("spans(0 rows) returned %d keys and %d values, want none",
			len(keys), len(values))
	}

	// One row is still one span, so the zero case is not simply "always empty".
	keys, values, err = cache.spans(0, 0, 1)
	if err != nil {
		t.Fatalf("spans(1 row): %v", err)
	}
	if len(keys) != 1 || len(values) != 1 {
		t.Errorf("spans(1 row) returned %d keys and %d values, want one each",
			len(keys), len(values))
	}
}
