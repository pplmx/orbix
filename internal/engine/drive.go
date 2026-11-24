package engine

import "sync"

// Drive pumps the engine until every submitted sequence has finished, while a caller-supplied
// function consumes each sequence's events concurrently.
//
// The shape this replaces was written out by hand five times in cmd/orbix, and every copy had to
// get the same three things right:
//
//   - each sequence's channel must be drained by a goroutine, because the pump produces the
//     events: reading first would wait for a token that cannot arrive, and reading after pumping
//     would never start, since Pump only returns once the sequences are done;
//   - the drains have to run concurrently with Pump rather than between rounds, because each
//     sequence's buffer is bounded at 64 events -- measured before this was fixed in `bench`,
//     with --max-tokens 65 against an idle fixture, every run generated exactly 64 tokens,
//     printed no finish reason and said nothing about why;
//   - Pump must return zero before the drains are joined, or a drainer blocks forever on a
//     channel that will not close.
//
// Getting any of the three wrong produces either a hang or a silently truncated run, which is
// why this belongs in one place rather than in four.
//
// consume is called once per sequence with that sequence's channel. It may count, print or
// collect, and it is the only part the callers actually differ in. It must not block on
// anything but the channel: the rounds are what produce events, and they cannot start until
// every drainer is running.
//
// Errors are the caller's to handle rather than Drive's, because what to do with one differs:
// `bench` aborts the whole measurement, while a scenario that counts early stops treats a normal
// finish as data. consume therefore returns nothing and reports through whatever it closes over.
//
// There is no cancellation parameter. These callers run to completion by construction --
// bench and the suite measure a fixed workload -- and a cancelled variant would have to invent
// an answer for the events still queued. The server's pump is a different animal (it polls and
// backs off while idle) and keeps its own loop.
func Drive(eng *Engine, chans []<-chan Event, consume func(ch <-chan Event)) {
	var wg sync.WaitGroup
	for _, ch := range chans {
		wg.Go(func() { consume(ch) })
	}
	for eng.Pump() > 0 {
	}
	wg.Wait()
}
