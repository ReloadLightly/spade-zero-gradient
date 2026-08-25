"""Validity gates (per-environment) and run-level smoke gates R1–R5.

D-02 (accepted 2026-08-25): floored hint-based regret stays the ONLY selective
fitness. The password-environment degenerate optimum is blocked by HARD
validity gates, not by reward blending:

  V1  executability battery passes (env_api.run_battery)
  V2  hint legality: <= 600 chars, no 'ACTION:' line, no fenced code
  V3  fitness eligibility band: no-hint win rate in [0.05, 0.95]
      (an env unsolvable without the hint, or trivially solved, contributes
      no fitness — it is logged but excluded)

Gates are not fitness. They never add reward; they only decide eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .env_api import run_battery

HINT_MAX_CHARS = 600
BAND_LO, BAND_HI = 0.05, 0.95
LEARNABLE_LO, LEARNABLE_HI = 0.20, 0.80   # descriptive metric, mirrors SPADE


@dataclass
class GateReport:
    v1_battery: bool
    v2_hint: bool
    v3_band: bool | None            # None until the env has been evaluated
    issues: list[str] = field(default_factory=list)

    @property
    def valid_for_fitness(self) -> bool:
        return self.v1_battery and self.v2_hint and bool(self.v3_band)


def check_hint(hint: str) -> list[str]:
    issues = []
    if not hint or not hint.strip():
        issues.append("V2: empty hint")
        return issues
    if len(hint) > HINT_MAX_CHARS:
        issues.append(f"V2: hint exceeds {HINT_MAX_CHARS} chars")
    if "ACTION:" in hint:
        issues.append("V2: hint contains literal 'ACTION:' line (action injection)")
    if "```" in hint:
        issues.append("V2: hint contains fenced code")
    return issues


def pre_eval_gates(code: str, hint: str) -> GateReport:
    battery = run_battery(code)
    hint_issues = check_hint(hint)
    return GateReport(
        v1_battery=battery.passed,
        v2_hint=not hint_issues,
        v3_band=None,
        issues=list(battery.issues) + hint_issues,
    )


def apply_band_gate(report: GateReport, win_nohint: float) -> GateReport:
    report.v3_band = BAND_LO <= win_nohint <= BAND_HI
    if not report.v3_band:
        report.issues.append(
            f"V3: no-hint win rate {win_nohint:.2f} outside [{BAND_LO}, {BAND_HI}]")
    return report


def is_learnable(win_nohint: float) -> bool:
    """SPADE's descriptive 'learnable band' (win rate 20–80%)."""
    return LEARNABLE_LO <= win_nohint <= LEARNABLE_HI
