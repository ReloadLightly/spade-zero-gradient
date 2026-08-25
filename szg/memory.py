"""Environment memory — the in-context substitute for designer weight updates.

JSONL of every generated env with measured outcomes. The designer (conditions
C1/C2) receives a digest: high-regret exemplars to build on, too-easy and
too-hard negatives to avoid, plus recent concepts for diversity pressure.
"""

from __future__ import annotations

import json
from pathlib import Path


class EnvMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def digest(self, k_high: int = 4, k_easy: int = 2, k_hard: int = 2,
               k_recent: int = 6) -> str:
        records = [r for r in self.all() if "win_nohint" in r]
        if not records:
            return "(environment memory is empty — this is the first round)"

        def line(r: dict) -> str:
            return (f"- [{r.get('skill', '?')}] {r.get('concept', r.get('env_name', '?'))} | "
                    f"no-hint win {r['win_nohint']:.2f}, hint win {r.get('win_hint', 0):.2f}, "
                    f"floored regret {r.get('floored_regret', 0):.2f}"
                    + ("" if r.get("valid_for_fitness", True) else
                       f" | INVALID: {'; '.join(r.get('gate_issues', [])[:2])}"))

        valid = [r for r in records if r.get("valid_for_fitness")]
        high = sorted(valid, key=lambda r: -r.get("floored_regret", 0))[:k_high]
        easy = [r for r in records if r["win_nohint"] > 0.95][:k_easy]
        hard = [r for r in records if r["win_nohint"] < 0.05][:k_hard]
        recent = records[-k_recent:]
        parts = ["ENVIRONMENT MEMORY (measured on the frozen solver):"]
        if high:
            parts.append("Highest-regret environments so far (the learning frontier — build on these):")
            parts += [line(r) for r in high]
        if easy:
            parts.append("Too easy (solver wins even without the hint — do NOT repeat this difficulty):")
            parts += [line(r) for r in easy]
        if hard:
            parts.append("Too hard / hint-gated (solver never wins without the hint — these are EXCLUDED from fitness):")
            parts += [line(r) for r in hard]
        parts.append("Most recent concepts (avoid duplicating them — diversity is measured):")
        parts += [f"- [{r.get('skill', '?')}] {r.get('concept', '?')}" for r in recent]
        return "\n".join(parts)
