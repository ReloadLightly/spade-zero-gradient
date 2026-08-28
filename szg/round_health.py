"""Is a completed round scientifically usable, or an infrastructure artifact?

cmd_round returns rc=0 even when every environment died at the designer
call: run_env_cycle catches a design failure per-env, records it, and the
round is then scored and logged like any other. On 2026-08-27 a usage-limit
outage produced two such rounds -- C0 r4 and C1 r4, both fitness 0.0 with
0/6 valid -- which were marked complete and would have entered the primary
endpoint as genuine zero-fitness rounds.

An env lost at the designer call is a MISSING MEASUREMENT, not a result. A
round containing any such loss is not comparable to a clean one, so the
driver retries it.

Exit 0 = clean, exit 1 = has infrastructure losses (caller should prune+retry).

    python -m szg.round_health runs/c0 C0 4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def infra_losses(out_dir: Path, condition: str, round_index: int) -> list[str]:
    prefix = f"{condition}_r{round_index:03d}_"
    lost = []
    p = out_dir / "envs.jsonl"
    if not p.exists():
        return lost
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
    return lost


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    out_dir, cond, r = Path(argv[1]), argv[2], int(argv[3])
    lost = infra_losses(out_dir, cond, r)
    if lost:
        print(f"UNUSABLE: {cond} round {r} lost {len(lost)} env(s) to backend "
              f"failure at the designer call: {', '.join(lost)}")
        return 1
    print(f"clean: {cond} round {r} has no infrastructure losses")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
