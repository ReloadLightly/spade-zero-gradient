import random


class TwoStageDriftEnv:
    def __init__(self):
        self.rng = None
        self.start = 0
        self.slope = 0
        self.period = 0
        self.bonus = 0
        self.probe_max = 12
        self.targets = {13: 0.3, 15: 0.3, 20: 0.4}
        self.resolved = {}
        self.steps = 0

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.start = self.rng.randint(1, 9)
        self.slope = self.rng.randint(2, 5)
        self.period = self.rng.choice([3, 4, 5])
        self.bonus = self.rng.randint(4, 9)
        self.resolved = {}
        self.steps = 0
        obs = (
            "GAME: Two-Stage Drift. A hidden integer sequence a[0], a[1], ... "
            "obeys: a[i] = START + i*SLOPE, except at every positive multiple "
            "of a hidden PERIOD (PERIOD is 3, 4, or 5), where a fixed BONUS is "
            "added on top. START, SLOPE, PERIOD, and BONUS are fixed constants "
            "you must discover.\n"
            "GOAL: correctly report a[13], a[15], and a[20] (worth 0.3, 0.3, "
            "and 0.4 reward respectively; total 1.0).\n"
            "ACTIONS: 'PROBE <i>' reveals the true value of a[i] for any i "
            "from 0 to 12. 'SUBMIT <i> <value>' locks in your answer for one "
            "of the three target indices (13, 15, 20); each target can only "
            "be graded once.\n"
            "You have 10 steps total. Every action (valid or not) counts as "
            "one step."
        )
        return obs, {}

    def _value(self, i):
        v = self.start + i * self.slope
        if i > 0 and i % self.period == 0:
            v += self.bonus
        return v

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False
        text = action.strip().split() if isinstance(action, str) else []

        if not text:
            obs = "Empty action. Use 'PROBE <i>' or 'SUBMIT <i> <value>'."
        else:
            cmd = text[0].upper()
            if cmd == "PROBE" and len(text) == 2:
                try:
                    i = int(text[1])
                except ValueError:
                    obs = "PROBE requires an integer index, e.g. 'PROBE 5'."
                else:
                    if 0 <= i <= self.probe_max:
                        obs = f"a[{i}] = {self._value(i)}."
                    elif i in self.targets:
                        obs = (
                            f"Index {i} is a held-out target, not probeable. "
                            "Use SUBMIT to answer it."
                        )
                    else:
                        obs = f"PROBE only works for indices 0 to {self.probe_max}."
            elif cmd == "SUBMIT" and len(text) == 3:
                try:
                    i = int(text[1])
                    guess = int(text[2])
                except ValueError:
                    obs = "SUBMIT requires two integers: 'SUBMIT <i> <value>'."
                else:
                    if i not in self.targets:
                        obs = f"SUBMIT index must be one of {sorted(self.targets)}."
                    elif i in self.resolved:
                        obs = (
                            f"Index {i} was already resolved "
                            f"(correct={self.resolved[i]}). No further credit."
                        )
                    else:
                        correct_val = self._value(i)
                        is_correct = guess == correct_val
                        self.resolved[i] = is_correct
                        if is_correct:
                            reward = self.targets[i]
                            obs = f"Correct! a[{i}] = {correct_val}."
                        else:
                            obs = (
                                f"Incorrect guess for a[{i}]. That target is "
                                "now resolved with no credit."
                            )
            else:
                obs = "Malformed action. Use 'PROBE <i>' or 'SUBMIT <i> <value>'."

        if len(self.resolved) == len(self.targets):
            terminated = True
        elif self.steps >= 10:
            truncated = True

        obs = f"{obs} Steps used: {self.steps}/10."
        return obs, reward, terminated, truncated, {}
