import random
import re


class FestivalCycleEnv:
    """Infer a hidden repeating festival pattern over a calendar, then
    predict festival days in an unseen future window."""

    PROBE_LO, PROBE_HI = 1, 40
    TARGET_LO, TARGET_HI = 41, 60
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.period = None
        self.offsets = None
        self.step_count = 0
        self.period_guessed = False
        self.done = False

    def _is_festival(self, day):
        return ((day - 1) % self.period) in self.offsets

    def _true_target_set(self):
        return {d for d in range(self.TARGET_LO, self.TARGET_HI + 1)
                if self._is_festival(d)}

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.period = self.rng.randint(3, 7)
        k = self.rng.randint(1, min(3, self.period - 1) if self.period > 1 else 1)
        self.offsets = set(self.rng.sample(range(self.period), k))
        self.step_count = 0
        self.period_guessed = False
        self.done = False

        obs = (
            "FESTIVAL CALENDAR PATTERN. A repeating festival cycle governs "
            f"days 1-{self.TARGET_HI} of an ancient calendar; you cannot see "
            "the rule directly.\n"
            f"You have {self.MAX_STEPS} actions total. Available actions:\n"
            f"  CHECK <day>      - reveals whether <day> (an integer in "
            f"{self.PROBE_LO}-{self.PROBE_HI}) is a festival day.\n"
            "  PERIOD <n>       - one-time guess at the cycle's repeat "
            "length; correct guess earns 0.3 reward.\n"
            f"  PREDICT <d1,d2,...> - final action, ends the episode. List "
            f"every day in {self.TARGET_LO}-{self.TARGET_HI} you believe is "
            "a festival day; an exact match earns 0.7, partial matches earn "
            "partial credit.\n"
            "Malformed actions cost a step but earn no reward."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already ended.", 0.0, True, False, {}

        self.step_count += 1
        action = (action or "").strip()

        m = re.fullmatch(r'CHECK\s+(-?\d+)', action, re.IGNORECASE)
        if m:
            day = int(m.group(1))
            if not (self.PROBE_LO <= day <= self.PROBE_HI):
                obs = (
                    f"Malformed CHECK: day must be between {self.PROBE_LO} "
                    f"and {self.PROBE_HI}."
                )
                return self._finish(obs, 0.0)
            fest = self._is_festival(day)
            obs = f"Day {day}: {'FESTIVAL' if fest else 'ordinary day'}."
            return self._finish(obs, 0.0)

        m = re.fullmatch(r'PERIOD\s+(-?\d+)', action, re.IGNORECASE)
        if m:
            if self.period_guessed:
                obs = "You already used your one-time period guess."
                return self._finish(obs, 0.0)
            self.period_guessed = True
            guess = int(m.group(1))
            if guess == self.period:
                obs = "Correct: that is the cycle's repeat length."
                return self._finish(obs, 0.3)
            obs = "Incorrect period guess. No further period guesses remain."
            return self._finish(obs, 0.0)

        m = re.fullmatch(r'PREDICT\s+(.*)', action, re.IGNORECASE)
        if m:
            raw = m.group(1)
            nums = re.findall(r'-?\d+', raw)
            predicted = {int(n) for n in nums}
            true_set = self._true_target_set()
            union = predicted | true_set
            if not union:
                jaccard = 1.0
            else:
                jaccard = len(predicted & true_set) / len(union)
            reward = 0.7 if predicted == true_set else 0.7 * jaccard
            obs = (
                f"PREDICT submitted: {sorted(predicted)}. True festival days "
                f"in {self.TARGET_LO}-{self.TARGET_HI}: {sorted(true_set)}. "
                f"Match score: {jaccard:.2f}."
            )
            self.done = True
            return obs, reward, True, False, {}

        obs = (
            "Malformed action. Use 'CHECK <day>', 'PERIOD <n>', or "
            "'PREDICT <d1,d2,...>'."
        )
        return self._finish(obs, 0.0)

    def _finish(self, obs, reward):
        if self.step_count >= self.MAX_STEPS:
            self.done = True
            obs += " Step limit reached; episode truncated without a PREDICT."
            return obs, reward, False, True, {}
        return obs, reward, False, False, {}
