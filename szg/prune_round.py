"""Remove all records for one (condition, round) from a run directory.

A round that dies partway -- e.g. the transient `claude -p rc=1` backend
outage that killed all three S3 conditions at round 2 on 2026-08-26 --
leaves orphaned env/episode/memory records behind. The driver re-runs the
round from scratch, which would append a second set of records with the
same env_ids. This strips the partial set first so the retry is clean.

Only touches append-only JSONL logs; never touches rounds.jsonl, which is
written once per round and so is absent for a failed round by construction.

    python -m szg.prune_round runs/c0 C0 2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def prune(out_dir: Path, condition: str, round_index: int) -> dict:
    prefix = f"{condition}_r{round_index:03d}_"
    removed = {}
    for name in ("envs.jsonl", "episodes.jsonl", "memory.jsonl"):
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
            if str(rec.get("env_id", "")).startswith(prefix):
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
