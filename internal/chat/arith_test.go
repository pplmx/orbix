package chat

import (
	"strings"
	"testing"
)

// TestArithmeticSubtraction covers the operator with the overflow check.
//
// The check exists because a template's operands are arbitrary text from the message list, not
// the small indices a template author had in mind. Measured before it: "-9223372036854775808"
// minus 1 rendered "9223372036854775807", and MaxInt64 minus -1 wrapped the other way. Both are
// numbers the template did not ask for appearing in a prompt.
func TestArithmeticSubtraction(t *testing.T) {
	cases := []struct {
		left, right, want string
	}{
		{"5", "3", "2"},
		{"3", "5", "-2"},
		{"0", "0", "0"},
		// A negative operand on the right is an addition, and must not be mistaken for the
		// "negative operands are refused" rule that belongs to modulo.
		{"5", "-3", "8"},
	}
	for _, c := range cases {
		v, err := evalArithmetic("-", c.left, c.right)
		if err != nil {
			t.Errorf("%s - %s: %v", c.left, c.right, err)
			continue
		}
		if v.text != c.want {
			t.Errorf("%s - %s = %q, want %q", c.left, c.right, v.text, c.want)
		}
	}
}

// TestArithmeticSubtractionRefusesOverflow pins the two values the comment records.
//
// Both directions matter: an underflow at MinInt64 and an overflow at MaxInt64 are the same
// defect seen from either end, and a guard written for one usually misses the other.
func TestArithmeticSubtractionRefusesOverflow(t *testing.T) {
	cases := []struct{ left, right string }{
		{"-9223372036854775808", "1"}, // MinInt64 - 1
		{"9223372036854775807", "-1"}, // MaxInt64 - (-1)
	}
	for _, c := range cases {
		_, err := evalArithmetic("-", c.left, c.right)
		if err == nil {
			t.Errorf("%s - %s was accepted; it wraps", c.left, c.right)
			continue
		}
		// The message names both operands, because "overflow" alone leaves a reader unable to
		// see which value was out of range.
		for _, want := range []string{c.left, c.right} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("the refusal does not name %s: %v", want, err)
			}
		}
	}
	// The boundaries themselves are representable and must be accepted: a guard that refused
	// them would break MaxInt64 - 0.
	if _, err := evalArithmetic("-", "9223372036854775807", "0"); err != nil {
		t.Errorf("MaxInt64 - 0 was refused: %v", err)
	}
	if _, err := evalArithmetic("-", "-9223372036854775808", "0"); err != nil {
		t.Errorf("MinInt64 - 0 was refused: %v", err)
	}
}

// TestArithmeticModulo covers the parity use the package documents.
func TestArithmeticModulo(t *testing.T) {
	cases := []struct {
		left, right, want string
	}{
		{"7", "3", "1"},
		{"6", "3", "0"},
		{"1", "1", "0"},
		{"0", "5", "0"},
	}
	for _, c := range cases {
		v, err := evalArithmetic("%", c.left, c.right)
		if err != nil {
			t.Errorf("%s %% %s: %v", c.left, c.right, err)
			continue
		}
		if v.text != c.want {
			t.Errorf("%s %% %s = %q, want %q", c.left, c.right, v.text, c.want)
		}
	}
}

// TestArithmeticModuloRefusesTheCasesWhereGoAndJinjaDisagree covers both refusals.
//
// Modulo by zero would panic; a negative operand would give a different answer here than in
// Jinja, and a wrong number in a prompt is worse than an error that says the case is unsupported.
func TestArithmeticModuloRefusesTheCasesWhereGoAndJinjaDisagree(t *testing.T) {
	cases := []struct {
		name, left, right, wants string
	}{
		{"modulo by zero", "5", "0", "zero"},
		{"negative dividend", "-5", "3", "negative"},
		{"negative divisor", "5", "-3", "negative"},
		{"both negative", "-5", "-3", "negative"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := evalArithmetic("%", c.left, c.right)
			if err == nil {
				t.Fatalf("%s %% %s was accepted", c.left, c.right)
			}
			if !strings.Contains(err.Error(), c.wants) {
				t.Errorf("the refusal does not say %q: %v", c.wants, err)
			}
		})
	}
}

// TestArithmeticRefusesNonNumericOperands covers the parse failure.
//
// A template's operand is text, so this is the common mistake rather than an exotic one: a
// variable holding "abc" reaching an arithmetic expression.
func TestArithmeticRefusesNonNumericOperands(t *testing.T) {
	for _, c := range [][2]string{{"abc", "1"}, {"1", "abc"}, {"", "1"}} {
		_, err := evalArithmetic("-", c[0], c[1])
		if err == nil {
			t.Errorf("%q - %q was accepted", c[0], c[1])
			continue
		}
		// Both operands, quoted: the reader has to see which one was not a number, and an
		// empty operand is otherwise invisible in the message.
		for _, want := range []string{`"` + c[0] + `"`, `"` + c[1] + `"`} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("the refusal does not name %s: %v", want, err)
			}
		}
	}
}

// TestArithmeticRefusesAnUnknownOperator keeps the default arm honest.
//
// Only - and % are implemented, and a template using * must be told rather than given a value.
func TestArithmeticRefusesAnUnknownOperator(t *testing.T) {
	_, err := evalArithmetic("*", "2", "3")
	if err == nil {
		t.Fatal("* was accepted")
	}
	if !strings.Contains(err.Error(), "*") {
		t.Errorf("the refusal does not name the operator: %v", err)
	}
}
