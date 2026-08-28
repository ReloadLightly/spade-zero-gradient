import random
import re


class LeapCycleWatchEnv:
    """Infer a hidden periodic leap-year rule from probed weekdays and extrapolate to unprobed years."""

    PROBE_MIN = 1
    PROBE_MAX = 24
    PROBE_BUDGET = 6
    TARGET_MIN = 30
    TARGET_MAX = 70
    MAX_STEPS = 10
    WEEKDAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    def __init__(self):
        self.rng = None
        self.period = None
        self.offset = None
        self.w0 = None
        self.weekday = {}
        self.target_years = []
        self.steps = 0
        self.probes_used = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.period = self.rng.choice([3, 4, 5, 6])
        self.offset = self.rng.randrange(self.period)
        self.w0 = self.rng.randrange(7)
        self.steps = 0
        self.probes_used = 0
        self.done = False

        self.weekday = {0: self.w0}
        w = self.w0
        for y in range(1, self.TARGET_MAX + 1):
            is_leap = (y - self.offset) % self.period == 0
            w = (w + (2 if is_leap else 1)) % 7
            self.weekday[y] = w

        self.target_years = sorted(
            self.rng.sample(range(self.TARGET_MIN, self.TARGET_MAX + 1), 4)
        )

        obs = (
            "YEAR-CYCLE WATCH: In this land, Year 0's Jan 1 fell on weekday "
            f"{self.w0} ({self.WEEKDAY_NAMES[self.w0]}). Weekdays are numbered "
            "0=Sun,1=Mon,2=Tue,3=Wed,4=Thu,5=Fri,6=Sat. Each subsequent year is "
            "either a common year (Jan 1's weekday advances by 1 from the previous "
            "year) or a leap year (advances by 2), governed by a hidden periodic "
            "rule. Your goal: predict the Jan 1 weekday number for years "
            f"{self.target_years} (in that order).\n"
            f"You may PROBE any year from {self.PROBE_MIN} to {self.PROBE_MAX} "
            f"(budget: {self.PROBE_BUDGET} probes) to learn its weekday number. "
            "When ready, submit your final answer with GUESS w1 w2 w3 w4 (four "
            "weekday numbers 0-6, one per target year in order) — this ends the "
            "episode.\n"
            "Action format: 'PROBE <year>' or 'GUESS <w1> <w2> <w3> <w4>'. "
            f"You have {self.MAX_STEPS} steps total."
        )
        info = {"target_years": list(self.target_years)}
        return obs, info

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()

        probe_match = re.fullmatch(r"(?i)PROBE\s+(-?\d+)", action)
        guess_match = re.fullmatch(
            r"(?i)GUESS\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)", action
        )

        if probe_match:
            year = int(probe_match.group(1))
            if year < self.PROBE_MIN or year > self.PROBE_MAX:
                obs = (
                    f"Invalid probe: year must be between {self.PROBE_MIN} "
                    f"and {self.PROBE_MAX}."
                )
            elif self.probes_used >= self.PROBE_BUDGET:
                obs = f"Probe budget exhausted ({self.PROBE_BUDGET} used). Submit your GUESS."
            else:
                self.probes_used += 1
                w = self.weekday[year]
                obs = (
                    f"Year {year}'s Jan 1 fell on weekday {w} "
                    f"({self.WEEKDAY_NAMES[w]}). Probes used: "
                    f"{self.probes_used}/{self.PROBE_BUDGET}."
                )
            reward = 0.0
            terminated = False
        elif guess_match:
            guesses = [int(guess_match.group(i)) for i in range(1, 5)]
            if any(g < 0 or g > 6 for g in guesses):
                obs = "Invalid guess: all four weekday numbers must be in 0-6. Try again."
                reward = 0.0
                terminated = False
            else:
                correct = sum(
                    1 for y, g in zip(self.target_years, guesses) if self.weekday[y] == g
                )
                reward = 0.25 * correct
                obs = (
                    f"Final answer submitted: {guesses}. Correct: {correct}/4. "
                    f"Actual weekdays were {[self.weekday[y] for y in self.target_years]}."
                )
                terminated = True
                self.done = True
        else:
            obs = (
                "Malformed action. Use 'PROBE <year>' or "
                "'GUESS <w1> <w2> <w3> <w4>' with integers."
            )
            reward = 0.0
            terminated = False

        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            self.done = True

        info = {"steps": self.steps, "probes_used": self.probes_used}
        return obs, reward, terminated, truncated, info
