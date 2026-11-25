package server

import (
	"fmt"
	"slices"
)

// maxBestOf is the largest candidate count this server accepts.
//
// best_of candidates are generated and then mostly discarded, so it multiplies a request's cost
// exactly as n does, and it is bounded by the same reasoning: every candidate is a sequence
// holding its own KV blocks, all in flight at once. The bound is also a floor for good sense
// -- past a few dozen candidates the scores are all within noise of each other.
const maxBestOf = maxChoices

// checkBestOf validates the candidate count, given how many completions were asked for.
//
// Two things are refused, each for a different reason:
//
//   - a negative or over-cap count is a malformed value, reported with the number that arrived;
//   - a count that is not greater than n is refused because OpenAI documents best_of as the
//     candidate count and n as how many to return, so best_of <= n asks for no selection at
//     all. Answering it by returning the n it asked for would silently ignore the field.
//
// The relationship check is skipped for a count of one or zero, which are the values that mean
// "no selection": a caller sending best_of: 1 alongside n: 1 is mid-migration from another
// server, and refusing that would fail a request this server answers exactly as asked.
func checkBestOf(bestOf, n int) (string, bool) {
	if bestOf < 0 {
		return fmt.Sprintf("best_of must not be negative, got %d", bestOf), true
	}
	if bestOf > maxBestOf {
		return fmt.Sprintf("best_of must be at most %d, got %d", maxBestOf, bestOf), true
	}
	if bestOf <= 1 {
		return "", false
	}
	if bestOf <= n {
		return fmt.Sprintf("best_of is the number of candidates and n is how many to return, "+
			"so best_of must be greater than n; got best_of %d and n %d", bestOf, n), true
	}
	return "", false
}

// rankCandidates returns the indices of the best keep candidates, best first.
//
// Ties break by index, so the same request returns the same choices in the same order: a sort
// left to compare equal scores would depend on the slice's starting order, which is stable
// today and would stop being obvious the moment the collection changes.
//
// A candidate with no per-token score -- no log probabilities, or none generated -- sorts last
// rather than being dropped. Dropping it would let a shorter response come back than the
// caller asked for, which is the failure the candidate generation exists to avoid.
func rankCandidates(results []choiceResult, shown []int, keep int) []int {
	order := make([]int, len(results))
	for i := range order {
		order[i] = i
	}
	scores := make([]float64, len(results))
	for i := range results {
		scores[i] = results[i].scoreShown(shown[i])
	}
	slices.SortStableFunc(order, func(a, b int) int {
		switch {
		case scores[a] > scores[b]:
			return -1
		case scores[a] < scores[b]:
			return 1
		}
		return a - b
	})
	if keep > len(order) {
		keep = len(order)
	}
	return order[:keep]
}

// shownTokens is how many of a candidate's tokens the response will report.
//
// It exists because a stop string cuts the reported log probabilities to the text that survived,
// and the ranking has to use the same set. Without it a candidate that generated thirty tokens
// after its stop string is ranked on all of them while the caller sees only the first few.
func (s *Server) shownTokens(c *choiceResult, p generateParams) int {
	text := s.tok.DecodeCompletion(c.out)
	if cut, found := cutAtStop(text, p.stop); found {
		text = cut
	}
	described := s.describeTokens(c.samples)
	keep, _ := keptTokens(described, len(text))
	return keep
}
