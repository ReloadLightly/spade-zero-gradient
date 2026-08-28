"""Is a completed round scientifically usable, or an infrastructure artifact?

cmd_round returns rc=0 even when every environment died at the designer
call: run_env_cycle catches a design failure per-env, records it, and the
round is then scored and logged like any other. On 2026-08-27 a usage-limit
outage produced two such rounds -- C0 r4 and C1 r4, both fitness 0.0 with
0/6 valid -- which were marked complete and would have entered the primary
endpoint as genuine zero-fitness rounds.

An env lost at the designer call is a MISSING MEASUREMENT, not a result.
But the threshold matters. The first version of this check rejected a round
containing ANY such loss, which livelocks: C2's round 4 deterministically
re-selects strategy S3 (verified by replaying select_parent with
Random(2004)), and the same seed yields the same topics and byte-identical
designer prompts, so a retry reproduces the same timeouts indefinitely.

The line that matches the registered semantics is PREREG §4's MIN_VALID: a
round scores only if at least 2 environments are valid for fitness. So a
round is UNUSABLE when infrastructure loss is what pushed it below that
floor -- its fitness 0.0 is fabricated rather than measured. A round that
still reached MIN_VALID is degraded, not fabricated: it carries real
measurements, its losses are logged per-env, and S4 reports them.

Exit 0 = clean, exit 1 = has infrastructure losses (caller should prune+retry).

    python -m szg.round_health runs/c0 C0 4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


MIN_VALID = 2   # mirrors szg.fitness.MIN_VALID (PREREG §4)


def round_stats(out_dir: Path, condition: str, round_index: int) -> tuple[list[str], int]:
    """(envs lost at the designer call, envs valid for fitness)."""
    prefix = f"{condition}_r{round_index:03d}_"
    lost, n_valid = [], 0
    p = out_dir / "envs.jsonl"
    if not p.exists():
        return lost, n_valid
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not str(rec.get("env_id", "")).startswith(prefix):
            continue
        # stage == "design" means the designer call itself never returned:
        # backend error or timeout. Never a statement about designer quality.
        if rec.get("stage") == "design":
            lost.append(rec["env_id"])
        if rec.get("valid_for_fitness"):
            n_valid += 1
    return lost, n_valid


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    out_dir, cond, r = Path(argv[1]), argv[2], int(argv[3])
    lost, n_valid = round_stats(out_dir, cond, r)
    if lost and n_valid < MIN_VALID:
        print(f"UNUSABLE: {cond} round {r} lost {len(lost)} env(s) to backend "
              f"failure at the designer call and reached only {n_valid} valid "
              f"env(s) (< MIN_VALID={MIN_VALID}); its fitness would be fabricated")
        return 1
    if lost:
        print(f"degraded but usable: {cond} round {r} lost {len(lost)} env(s) to "
              f"backend failure, still reached {n_valid} valid env(s)")
        return 0
    print(f"clean: {cond} round {r} has no infrastructure losses")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
