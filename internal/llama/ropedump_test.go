package llama

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestDumpRope writes rotated heads for scripts/rope_xcheck.py to compare.
//
// The Go tests pin rope against values computed from the same reading of the formula, which is
// what a same-source check can do. The script rotates the same input with numpy using the
// definition from the RoFormer paper -- adjacent pairs for the normal layout, half-offset pairs
// for NEOX -- so a mistake in the pairing or in the frequency geometric is visible.
//
// Skipped unless ORBIX_ROPEDUMP names an output file.
func TestDumpRope(t *testing.T) {
	out := os.Getenv("ORBIX_ROPEDUMP")
	if out == "" {
		t.Skip("set ORBIX_ROPEDUMP to write a comparison corpus")
	}

	type caseOut struct {
		HeadDim int       `json:"head_dim"`
		RopeDim int       `json:"rope_dim"`
		Base    float64   `json:"base"`
		Scale   float64   `json:"scale"`
		Layout  string    `json:"layout"`
		Pos     []int     `json:"pos"`
		X       []float32 `json:"x"`
		Y       []float32 `json:"y"`
	}

	// Two layouts, two head widths, and a non-default base and scale: the first three vary the
	// pairing, the last two the frequency.
	specs := []struct {
		name    string
		headDim int
		ropeDim int
		base    float64
		scale   float64
		layout  RopeLayout
	}{
		{"normal-64", 64, 64, 10000, 1, RopeNormal},
		{"neox-64", 64, 64, 10000, 1, RopeNeox},
		{"normal-128-partial", 128, 64, 10000, 1, RopeNormal},
		{"neox-128-partial", 128, 64, 10000, 1, RopeNeox},
		{"normal-base500k", 64, 64, 500000, 1, RopeNormal},
		{"neox-scale-0.5", 64, 64, 10000, 0.5, RopeNeox},
	}

	var got []caseOut
	for _, s := range specs {
		cfg := Config{
			HeadDim: s.headDim, RopeDim: s.ropeDim, RopeBase: float32(s.base),
			RopeScale: float32(s.scale), RopeLayout: s.layout,
		}
		r, err := newRope(cfg, 16)
		if err != nil {
			t.Fatalf("%s: %v", s.name, err)
		}
		x := make([]float32, s.headDim)
		for i := range x {
			// A deterministic ramp with both signs, so a rotation that is off by a
			// quadrant or a sign shows as a difference rather than cancelling.
			x[i] = float32(math.Sin(float64(i)*0.7+0.3)) * 2
		}
		positions := []int{0, 1, 5, 15}
		orig := append([]float32(nil), x...)
		ys := make([]float32, 0, len(positions)*s.headDim)
		for _, pos := range positions {
			xx := append([]float32(nil), orig...)
			if err := r.apply(xx, pos); err != nil {
				t.Fatalf("%s at %d: %v", s.name, pos, err)
			}
			ys = append(ys, xx...)
		}
		got = append(got, caseOut{
			HeadDim: s.headDim, RopeDim: s.ropeDim, Base: s.base, Scale: s.scale,
			Layout: map[RopeLayout]string{RopeNormal: "normal", RopeNeox: "neox"}[s.layout],
			Pos:    positions, X: orig, Y: ys,
		})
	}

	b, err := json.Marshal(map[string]any{"cases": got})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(out, b, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
}
