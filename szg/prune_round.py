"""Remove all records for one (condition, round) from a run directory.

A round that dies partway -- e.g. the transient `claude -p rc=1` backend
outage that killed all three S3 conditions at round 2 on 2026-08-26 --
leaves orphaned env/episode/memory records behind. The driver re-runs the
round from scratch, which would append a second set of records with the
same env_ids. This strips the partial set first so the retry is clean.

Strips rounds.jsonl too. An earlier version of this tool deliberately left
rounds.jsonl alone, on the assumption that a failed round never gets one
written. That assumption is FALSE for C2: cmd_round writes rounds.jsonl and
records the strategy fitness BEFORE the --mutate step, so a round whose
mutate call fails is scored, logged, and then reported as rc=1. On
2026-08-27 a usage-limit outage produced exactly that -- C2 round 4 lost all
six envs at the designer call, was scored INVALID_ROUND with fitness 0.0,
had its rounds.jsonl entry written, then failed at mutate. Pruning only the
env records left a phantom round in the trajectory that looked like a real
zero-fitness round.

    python -m szg.prune_round runs/c0 C0 2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def prune(out_dir: Path, condition: str, round_index: int) -> dict:
    prefix = f"{condition}_r{round_index:03d}_"
    removed = {}
    for name in ("envs.jsonl", "episodes.jsonl", "memory.jsonl", "rounds.jsonl"):
        p = out_dir / name
        if not p.exists():
            continue
        kept, dropped = [], 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)      # keep anything unparseable rather than lose it
                continue
            if name == "rounds.jsonl":
                match = (rec.get("condition") == condition
                         and rec.get("round_index") == round_index)
            else:
                match = str(rec.get("env_id", "")).startswith(prefix)
            if match:
                dropped += 1
            else:
                kept.append(line)
        p.write_text("".join(l + "\n" for l in kept))
        removed[name] = dropped
    return removed


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    out_dir, condition, round_index = Path(argv[1]), argv[2], int(argv[3])
    removed = prune(out_dir, condition, round_index)
    print(f"pruned {condition} round {round_index} from {out_dir}: {removed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
