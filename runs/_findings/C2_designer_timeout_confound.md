# Finding — C2 loses environments to designer timeouts (directional confound)

Recorded 2026-08-27, mid-S3, rounds 3–4.

## What happened

Four C2 environments never reached evaluation. Their records carry
`stage: "design"` and:

    BackendError("backend failed after retries: Command
    '['claude','-p','--model','sonnet','--output-format','text']'
    timed out after ...")

These are **timeouts** against `backends.py`'s `timeout_s = 240.0`, not the
transient `rc=1` outage that killed round 2. This log initially recorded them
as backend outage; that was wrong and is corrected here.

## The distribution is not random

    envs lost at the designer call        envs reaching evaluation
      c0  0 / 24                            c0  21/24  (88%)
      c1  0 / 27                            c1  25/27  (93%)
      c2  4 / 27                            c2  21/27  (78%)

C2 is the only condition whose designer prompt carries BOTH an accumulating
memory digest AND an evolved strategy text. C0 carries neither and has never
timed out. C1 carries a digest but no evolved strategy and has never timed
out either.

## Why it matters

The loss is directional and lands on the condition the experiment is about.
C2 is penalised for carrying more context — which is the very mechanism
under test. A designer whose prompt grows as it accumulates memory and
evolves its strategy will systematically lose environments to a fixed
timeout, and those losses look like lower productivity rather than an
infrastructure limit.

It is also concentrated in rounds 3 and 4, i.e. inside the final third the
registered primary endpoint (§5) measures.

Interaction with H-RULE-1 (information parity): the rule requires that no
condition sees vocabulary another lacks. Nothing here violates that as
written — all conditions see the full contract — but the timeout imposes an
*implicit* budget that binds only on the condition with the most context.
That asymmetry is worth stating in the writeup even though no rule forbids
it.

## Why nothing was patched mid-grid

Raising `timeout_s` at round 5 would make one round run on different
infrastructure from rounds 0–4, replacing a documented constant bias with a
bias correlated with round index. That is worse for a trajectory analysis
and would buy one round. The grid finished as configured.

## Proposed fix (for Roland, at the S4 STOP)

Re-run C2 in full with a larger `timeout_s`, or cap the designer prompt
(digest truncation) so prompt length cannot grow without bound. A full
re-run gives a clean C2 arm rather than a patchwork of rounds.

`timeout_s` is infrastructure, not a registered parameter: a timeout means
the measurement failed, not that the designer did. No PREREG amendment is
required — only a decision on the compute.

Until then, **C2's fitness numbers should be read as a lower bound**, and
the realised env count per condition reported alongside them.
