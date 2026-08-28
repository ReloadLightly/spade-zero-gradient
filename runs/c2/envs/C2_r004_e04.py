import random
import re


class PlanetaryConjunctionEnv:
    def __init__(self):
        self.R = 60
        self.candidates = [72, 78, 84, 90, 96, 102]
        self.max_probe_day = 90
        self.max_steps = 10
        self.after_day = 200

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.P = self.rng.choice(self.candidates)
        self.consistent = list(self.candidates)
        self.steps = 0
        self.milestones_hit = set()
        self.done = False
        obs = (
            f"You are tracking a mystery planet whose orbital period P (in days) is one of "
            f"{self.candidates}, while the reference planet's period is fixed at R={self.R} days. "
            f"Both planets start aligned at day 0. Send 'PROBE <day>' (integer 1-{self.max_probe_day}) "
            f"to learn the coarse angular-separation reading between the two planets on that day: "
            f"CONJUNCTION (near-aligned), NEAR, FAR, or OPPOSITION (near-opposite). Use readings "
            f"across different probe days to narrow down which period P is real. When ready, send "
            f"'COMMIT <day>' with your prediction for the day (integer > {self.after_day}) of the "
            f"NEXT conjunction after day {self.after_day}. COMMIT ends the episode. You have "
            f"{self.max_steps} total actions (probes + commit combined); malformed actions are "
            f"corrected but cost nothing."
        )
        return obs, {}

    def _bucket(self, P, day):
        rate = 360.0 * (1.0 / self.R - 1.0 / P)
        phi = (rate * day) % 360.0
        dist = min(phi, 360.0 - phi)
        if dist <= 15:
            return "CONJUNCTION"
        if dist <= 90:
            return "NEAR"
        if dist <= 150:
            return "FAR"
        return "OPPOSITION"

    def _next_conjunction(self, P):
        t = self.after_day + 1
        limit = self.after_day + 1000
        while t <= limit:
            if self._bucket(P, t) == "CONJUNCTION":
                return t
            t += 1
        return limit

    def step(self, action):
        if self.done:
            return "Episode already ended.", 0.0, True, False, {}

        action = (action or "").strip().upper()
        m_probe = re.match(r"^PROBE\s+(-?\d+)$", action)
        m_commit = re.match(r"^COMMIT\s+(-?\d+)$", action)

        if m_probe:
            day = int(m_probe.group(1))
            if day < 1 or day > self.max_probe_day:
                return (
                    f"Invalid PROBE day; must be 1-{self.max_probe_day}. No action consumed.",
                    0.0, False, False, {},
                )
            self.steps += 1
            reading = self._bucket(self.P, day)
            self.consistent = [c for c in self.consistent if self._bucket(c, day) == reading]
            after = len(self.consistent)
            reward = 0.0
            for threshold, bonus in ((4, 0.1), (3, 0.1), (2, 0.1)):
                if after <= threshold and threshold not in self.milestones_hit:
                    self.milestones_hit.add(threshold)
                    reward += bonus
            remaining = self.max_steps - self.steps
            obs = (
                f"Day {day}: {reading}. Consistent candidates remaining: {sorted(self.consistent)}. "
                f"Actions left: {remaining}."
            )
            if self.steps >= self.max_steps:
                self.done = True
                return obs + " Step limit reached without a COMMIT.", reward, False, True, {}
            return obs, reward, False, False, {}

        if m_commit:
            guess = int(m_commit.group(1))
            self.steps += 1
            true_day = self._next_conjunction(self.P)
            error = abs(guess - true_day)
            if error <= 5:
                reward = 0.7
            elif error <= 15:
                reward = 0.5
            elif error <= 30:
                reward = 0.3
            elif error <= 60:
                reward = 0.1
            else:
                reward = 0.0
            self.done = True
            obs = (
                f"COMMIT day {guess}. True next conjunction after day {self.after_day} was day "
                f"{true_day} (error {error}). The mystery planet's period was P={self.P}."
            )
            return obs, reward, True, False, {}

        return (
            "Malformed action. Use 'PROBE <day>' or 'COMMIT <day>' with an integer day. "
            "No action consumed."
        ), 0.0, False, False, {}
