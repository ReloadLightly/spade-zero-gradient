# Finding — A2 was implemented incompletely: the memory digest still uses the pre-A2 band

Recorded 2026-08-28. Surfaced by an independent review, verified here.

## The gap

Amendment A2 moved V3's eligibility band from `win_nohint` to `mean_nohint`.
I updated `gates.py`, both `cli.py` call sites, PREREG §4/§7 and the tests.
I did **not** update `szg/memory.py`, whose digest still segments
environments by the pre-A2 win-rate criterion:

    easy = [r for r in records if r["win_nohint"] > 0.95]
    hard = [r for r in records if r["win_nohint"] < 0.05]

I did inspect `memory.py` during the A1 work and cleared it — but only for
the `None`-formatting crash. I never asked whether its band logic needed the
A2 change too.

## Why it matters

The digest is the *learning signal* for C1 and C2 — the only channel through
which the frozen designer can improve. It is now inconsistent with the
fitness those conditions are selected on:

    C1: 9 of 23 memory rows are valid-for-fitness but fall outside the win band
    C2: 5 of 20 memory rows are valid-for-fitness but fall outside the win band

So environments that *do* contribute fitness — including each condition's
highest-regret frontier environments — can be presented to the designer in
the "too easy" / "too hard" framing, i.e. as things to avoid. The review
found a case where one environment appears under both "build on these" and
the excluded framing in the same digest.

Direction of bias: **against the treatment arms.** C0 has no memory and is
unaffected. C1 and C2 are the conditions being tested for improvement, and
they are the only ones receiving a self-contradictory signal.

## Not fixed mid-grid

Changing the digest changes the learning signal, which changes the
experiment. Rounds 0–3 ran under the current digest; patching now would
split every C1/C2 chain across two different treatments. The grid finishes
as configured.

## For Roland at the S4 STOP

1. The writeup must state that C1/C2 received a digest whose easy/hard
   segmentation predates A2, and that the bias runs against those arms.
2. Any re-run should align `memory.py` to the A2 band (a two-line change) —
   ideally in the same pass as the RNG-stream separation and `timeout_s`.
3. This is a second instance of the same failure mode as A1's evolve.py
   miss: an amendment applied to the obvious call sites but not to every
   consumer of the changed quantity. A re-run should grep for the changed
   variable across the package rather than trusting the diff.
