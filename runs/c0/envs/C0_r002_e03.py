import random
import re


class AuthenticGemBalanceEnv:
    """Find the one counterfeit gem among 8 via balance-scale weighings."""

    N_GEMS = 8
    MAX_STEPS = 10
    DIRECTIONS = ("HEAVIER", "LIGHTER")

    WEIGH_RE = re.compile(r'^WEIGH\s+([0-9,\s]+)\|([0-9,\s]+)$', re.IGNORECASE)
    GUESS_RE = re.compile(r'^GUESS\s+(\d+)\s+(HEAVIER|LIGHTER)$', re.IGNORECASE)

    def __init__(self):
        self.rng = None
        self.fake_index = None
        self.direction = None
        self.hypotheses = set()
        self.step_count = 0
        self.achieved = set()
        self.terminated = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.fake_index = self.rng.randint(1, self.N_GEMS)
        self.direction = self.rng.choice(self.DIRECTIONS)
        self.hypotheses = {
            (g, d) for g in range(1, self.N_GEMS + 1) for d in self.DIRECTIONS
        }
        self.step_count = 0
        self.achieved = set()
        self.terminated = False

        obs = (
            "You are appraising 8 gems, labeled 1-8. Exactly one is a counterfeit: "
            "it is either HEAVIER or LIGHTER than all 7 genuine gems (which are "
            "identical in weight to each other). You do not know which gem or which "
            "direction. You have a balance scale.\n\n"
            "Goal: identify the counterfeit's index AND whether it is HEAVIER or LIGHTER, "
            "within 10 total actions.\n\n"
            "Actions (exactly one per turn):\n"
            "  WEIGH a,b,c|d,e,f   -- place left-of-bar indices, then right-of-bar indices, "
            "separated by '|'. Both sides must have the same nonzero count of distinct, "
            "unused-elsewhere gems (no gem on both sides). Reply tells which side went down, "
            "or BALANCED.\n"
            "  GUESS <index> <HEAVIER|LIGHTER>  -- final answer; ends the episode.\n\n"
            "You have 16 initially-possible (gem, direction) hypotheses. Every weighing must "
            "be used to eliminate hypotheses inconsistent with its outcome."
        )
        return obs, {"n_gems": self.N_GEMS, "max_steps": self.MAX_STEPS}

    def _simulate_outcome(self, gem, direction, left, right):
        if gem in left:
            return "LEFT_DOWN" if direction == "HEAVIER" else "RIGHT_DOWN"
        if gem in right:
            return "RIGHT_DOWN" if direction == "HEAVIER" else "LEFT_DOWN"
        return "BALANCED"

    def _malformed(self, message):
        self.step_count += 1
        truncated = self.step_count >= self.MAX_STEPS
        if truncated:
            self.terminated = True
        obs = message + f" ({self.MAX_STEPS - self.step_count} actions remaining.)"
        return obs, 0.0, False, truncated, {"valid": False, "step": self.step_count}

    def _parse_group(self, text):
        parts = [p.strip() for p in text.split(",") if p.strip() != ""]
        return [int(p) for p in parts if p.isdigit()]

    def step(self, action):
        if self.terminated:
            return "Episode already ended.", 0.0, True, False, {"valid": False}

        action = (action or "").strip()
        upper = action.upper()

        if upper.startswith("WEIGH"):
            m = self.WEIGH_RE.match(action)
            if not m:
                return self._malformed(
                    "Malformed WEIGH. Use: WEIGH 1,2,3|4,5,6 (comma-separated indices, "
                    "'|' between the two pans)."
                )
            left = self._parse_group(m.group(1))
            right = self._parse_group(m.group(2))
            valid_range = all(1 <= g <= self.N_GEMS for g in left + right)
            no_overlap = len(set(left) & set(right)) == 0
            equal_nonzero = len(left) == len(right) and len(left) > 0
            distinct = len(set(left)) == len(left) and len(set(right)) == len(right)
            if not (valid_range and no_overlap and equal_nonzero and distinct):
                return self._malformed(
                    "Invalid weighing: both pans need equal, nonzero, non-overlapping "
                    "sets of distinct indices from 1-8."
                )

            self.step_count += 1
            outcome = self._simulate_outcome(self.fake_index, self.direction, left, right)
            before = len(self.hypotheses)
            self.hypotheses = {
                h for h in self.hypotheses
                if self._simulate_outcome(h[0], h[1], left, right) == outcome
            }
            after = len(self.hypotheses)

            reward = 0.0
            if after <= 6 and "mid" not in self.achieved:
                reward += 0.2
                self.achieved.add("mid")
            if after <= 2 and "narrow" not in self.achieved:
                reward += 0.3
                self.achieved.add("narrow")

            truncated = self.step_count >= self.MAX_STEPS
            if truncated:
                self.terminated = True

            obs = (
                f"Result: {outcome}. Hypotheses narrowed from {before} to {after}. "
                f"({self.MAX_STEPS - self.step_count} actions remaining.)"
            )
            if truncated:
                obs += " Out of actions -- episode over without a guess."
            return obs, reward, False, truncated, {
                "valid": True, "step": self.step_count, "hypotheses_remaining": after
            }

        if upper.startswith("GUESS"):
            m = self.GUESS_RE.match(action)
            if not m:
                return self._malformed(
                    "Malformed GUESS. Use: GUESS <index 1-8> <HEAVIER|LIGHTER>."
                )
            gem = int(m.group(1))
            direction = m.group(2).upper()
            if not (1 <= gem <= self.N_GEMS):
                return self._malformed("Guessed index must be between 1 and 8.")

            self.step_count += 1
            self.terminated = True
            correct = (gem == self.fake_index) and (direction == self.direction)
            reward = 0.5 if correct else 0.0
            if correct:
                obs = f"Correct! Gem {gem} was the counterfeit ({direction})."
            else:
                obs = (
                    f"Incorrect. Gem {gem} ({direction}) was not the counterfeit. "
                    "Episode over."
                )
            return obs, reward, True, False, {
                "valid": True, "step": self.step_count, "correct": correct
            }

        return self._malformed(
            "Unrecognized action. Use WEIGH a,b,c|d,e,f or GUESS <index> <HEAVIER|LIGHTER>."
        )
