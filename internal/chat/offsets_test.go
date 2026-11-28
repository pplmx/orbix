package chat

import (
	"strings"
	"testing"
)

// TestStatementOffsetsPointAtTheirTag pins the two offsets added when splitForIf and
// splitStatement started reporting their caller's position.
//
// Both helpers work on text cut out of a tag and have no offset of their own, so the caller wraps
// the error with the position it does have. The cases put text before the tag on purpose: a
// message that always said "at offset 0" would pass when the tag is at the start of the template
// and prove nothing.
func TestStatementOffsetsPointAtTheirTag(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want string
	}{
		{"bare if in for", "hello {% for x in y if %}{% endfor %}", "at offset 6"},
		{"empty statement", "abcdef{% %}x{% endif %}", "at offset 6"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := Parse(c.src)
			if err == nil {
				t.Fatalf("Parse(%q) accepted it", c.src)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("Parse(%q) error %q does not say %q", c.src, err, c.want)
			}
		})
	}
}
