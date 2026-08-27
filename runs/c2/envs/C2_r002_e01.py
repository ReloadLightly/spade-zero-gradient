import random


class GrooveDensityRampEnv:
    """Solver infers a hidden arithmetic hit-count ramp plus a hidden
    even-spacing/rotation rule for a kick-drum groove from two notated
    bars, may probe a hidden third bar for confirmation, then must
    predict specific steps of a fourth, never-shown bar."""

    def __init__(self):
        self.n = 16
        self.max_steps = 10

    def _hits_for_k(self, k):
        base = sorted(set((i * self.n) // k for i in range(k)))
        return set((off + self.r) % self.n for off in base)

    def _bar_string(self, hitset):
        return "".join("x" if s in hitset else "." for s in range(self.n))

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.k1 = self.rng.randint(2, 5)
        self.d = self.rng.choice([1, 2])
        self.r = self.rng.randint(0, self.n - 1)
        self.k = [self.k1 + i * self.d for i in range(4)]
        self.hits = [self._hits_for_k(k) for k in self.k]
        self.bar_strings = [self._bar_string(h) for h in self.hits]
        self.query_positions = sorted(self.rng.sample(range(self.n), 4))
        self.true_answers = [
            "H" if s in self.hits[3] else "R" for s in self.query_positions
        ]
        self.step_count = 0
        self.done = False

        obs = (
            "GROOVE DENSITY RAMP\n"
            f"A kick drum plays a {self.n}-step bar (steps 0-{self.n-1}, "
            "'x'=hit, '.'=rest). Two consecutive notated bars follow.\n"
            f"Bar 1: {self.bar_strings[0]}\n"
            f"Bar 2: {self.bar_strings[1]}\n"
            "Bar 3 is NOT shown. Bar 4 is NOT shown either, but you must "
            "answer whether specific Bar-4 steps are hits ('H') or rests "
            "('R'): steps "
            + ", ".join(str(s) for s in self.query_positions)
            + ".\n"
            "The groove follows a consistent hidden rule across all four "
            "bars — figure it out from Bar 1 and Bar 2.\n"
            "Actions (exactly one per turn):\n"
            "  PROBE <step>   - reveal Bar 3's true symbol at that step "
            "(0-15). Costs a turn, no reward.\n"
            "  ANSWER <v1> <v2> <v3> <v4> - your H/R prediction for the "
            "four Bar-4 query steps above, in the order listed. Ends the "
            "episode.\n"
            f"You have {self.max_steps} turns total (probes and the "
            "answer both count)."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()
        parts = text.split()

        if not parts:
            obs = "Empty action. Use 'PROBE <step>' or 'ANSWER <v1> <v2> <v3> <v4>'."
            truncated = self.step_count >= self.max_steps
            self.done = self.done or truncated
            return obs, 0.0, False, truncated, {}

        verb = parts[0].upper()

        if verb == "PROBE":
            if len(parts) != 2 or not parts[1].lstrip("-").isdigit():
                obs = "Malformed PROBE. Use 'PROBE <step>' with step an integer 0-15."
                truncated = self.step_count >= self.max_steps
                self.done = self.done or truncated
                return obs, 0.0, False, truncated, {}
            s = int(parts[1])
            if not (0 <= s < self.n):
                obs = f"Step must be in 0-{self.n-1}."
                truncated = self.step_count >= self.max_steps
                self.done = self.done or truncated
                return obs, 0.0, False, truncated, {}
            symbol = "x (hit)" if s in self.hits[2] else ". (rest)"
            obs = f"Bar 3, step {s}: {symbol}"
            truncated = self.step_count >= self.max_steps
            self.done = self.done or truncated
            return obs, 0.0, False, truncated, {}

        if verb == "ANSWER":
            vals = [p.upper() for p in parts[1:]]
            if len(vals) != 4 or any(v not in ("H", "R") for v in vals):
                obs = (
                    "Malformed ANSWER. Use 'ANSWER <v1> <v2> <v3> <v4>' "
                    "with each value exactly H or R, for the four query "
                    "steps in the order given at the start."
                )
                truncated = self.step_count >= self.max_steps
                self.done = self.done or truncated
                return obs, 0.0, False, truncated, {}
            correct = sum(1 for a, t in zip(vals, self.true_answers) if a == t)
            reward = correct * 0.2 + (0.2 if correct == 4 else 0.0)
            self.done = True
            obs = (
                f"ANSWER submitted: {' '.join(vals)}. "
                f"Correct: {correct}/4 (bonus {'earned' if correct == 4 else 'not earned'}). "
                f"True Bar 4 pattern: {self.bar_strings[3]}"
            )
            return obs, reward, True, False, {}

        obs = "Unknown action verb. Use 'PROBE <step>' or 'ANSWER <v1> <v2> <v3> <v4>'."
        truncated = self.step_count >= self.max_steps
        self.done = self.done or truncated
        return obs, 0.0, False, truncated, {}
