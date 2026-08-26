import random


class DrumRotationContinuationEnv:
    """Infer a hidden circular rotation shift between drum bars and extrapolate."""

    L = 8  # steps per bar (16th notes in a 2-beat cell)
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.done = False

        k = self.rng.randint(2, 4)
        hit_positions = set(self.rng.sample(range(self.L), k))
        self.bar1 = [i in hit_positions for i in range(self.L)]

        R = self.rng.randint(1, self.L - 1)
        tries = 0
        while self._rotate(self.bar1, R) == self.bar1 and tries < 10:
            R = self.rng.randint(1, self.L - 1)
            tries += 1
        self.R = R

        self.bar2 = self._rotate(self.bar1, self.R)
        self.bar3 = self._rotate(self.bar1, (2 * self.R) % self.L)
        self.bar4 = self._rotate(self.bar1, (3 * self.R) % self.L)

        reveal_count = 3
        self.revealed = set(self.rng.sample(range(self.L), reveal_count))

        self.answer1_given = False
        self.answer2_given = False

        obs = (
            "DRUM RHYTHM CONTINUATION.\n"
            f"Grid: {self.L} steps per bar (positions 1-{self.L}). X = hit, . = rest, ? = unknown.\n"
            f"Bar 1 (fully notated): {self._to_str(self.bar1)}\n"
            f"Bar 2 (partially notated): {self._masked_str()}\n"
            "RULE: every later bar is bar 1 rotated by a FIXED shift R (same R each step), "
            "cumulatively: bar2=shift(R), bar3=shift(2R), bar4=shift(3R), all mod "
            f"{self.L}.\n"
            "GOAL: determine R, then predict Bar 3 and Bar 4 exactly.\n"
            "ACTIONS:\n"
            "  PROBE <n>   - reveal the true symbol at step n (1-8) of Bar 2\n"
            "  ANSWER1 <s> - submit your 8-character X/. prediction for Bar 3\n"
            "  ANSWER2 <s> - submit your 8-character X/. prediction for Bar 4 (after ANSWER1)\n"
            f"You have {self.MAX_STEPS} steps total. Malformed actions cost a step but earn no reward."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        reward = 0.0
        terminated = False
        obs = ""

        parts = (action or "").strip().split(None, 1)
        cmd = parts[0].upper() if parts else ""

        if cmd == "PROBE":
            arg = parts[1].strip() if len(parts) > 1 else ""
            if not arg.isdigit() or not (1 <= int(arg) <= self.L):
                obs = f"Malformed PROBE. Use 'PROBE <n>' with n in 1-{self.L}."
            else:
                idx = int(arg) - 1
                self.revealed.add(idx)
                val = "X" if self.bar2[idx] else "."
                obs = f"Bar 2, step {idx + 1} = {val}. Bar 2 now: {self._masked_str()}"

        elif cmd == "ANSWER1":
            s = parts[1].strip().upper() if len(parts) > 1 else ""
            if len(s) != self.L or any(c not in "X." for c in s):
                obs = f"Malformed ANSWER1. Submit exactly {self.L} characters of X/."
            elif self.answer1_given:
                obs = "Bar 3 already answered. Use ANSWER2 for Bar 4."
            else:
                target = self._to_str(self.bar3)
                matches = sum(1 for a, b in zip(s, target) if a == b)
                frac = matches / self.L
                exact = s == target
                reward = 0.15 * frac + (0.35 if exact else 0.0)
                self.answer1_given = True
                obs = (
                    f"Bar 3 submitted: {s}. Matched {matches}/{self.L} steps."
                    + (" Exact match!" if exact else " Not exact.")
                    + " Now submit ANSWER2 for Bar 4."
                )

        elif cmd == "ANSWER2":
            s = parts[1].strip().upper() if len(parts) > 1 else ""
            if len(s) != self.L or any(c not in "X." for c in s):
                obs = f"Malformed ANSWER2. Submit exactly {self.L} characters of X/."
            elif not self.answer1_given:
                obs = "Submit ANSWER1 for Bar 3 before ANSWER2."
            elif self.answer2_given:
                obs = "Bar 4 already answered."
            else:
                target = self._to_str(self.bar4)
                matches = sum(1 for a, b in zip(s, target) if a == b)
                frac = matches / self.L
                exact = s == target
                reward = 0.15 * frac + (0.35 if exact else 0.0)
                self.answer2_given = True
                terminated = True
                obs = (
                    f"Bar 4 submitted: {s}. Matched {matches}/{self.L} steps."
                    + (" Exact match!" if exact else " Not exact.")
                    + " Episode complete."
                )

        else:
            obs = "Unknown action. Use PROBE <n>, ANSWER1 <8 chars>, or ANSWER2 <8 chars>."

        truncated = False
        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated = True
        if terminated or truncated:
            self.done = True

        return obs, reward, terminated, truncated, {}

    def _rotate(self, bits, shift):
        new = [False] * self.L
        for i in range(self.L):
            new[(i + shift) % self.L] = bits[i]
        return new

    def _to_str(self, bits):
        return "".join("X" if b else "." for b in bits)

    def _masked_str(self):
        return "".join(
            ("X" if self.bar2[i] else ".") if i in self.revealed else "?"
            for i in range(self.L)
        )
