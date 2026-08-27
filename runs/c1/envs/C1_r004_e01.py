import random
import re


class TwoStageLedgerEnv:
    MIN_N = 1
    MAX_N = 20
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.m = None
        self.b = None
        self.offsets = None
        self.held_out = None
        self.queried = None
        self.predicted = None
        self.steps = 0
        self.done = False

    def _value(self, n):
        return self.m * n + self.b + self.offsets[n % 3]

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.m = self.rng.choice([2, 3, 4])
        self.b = self.rng.randint(-6, 6)
        while True:
            offsets = [self.rng.randint(-4, 4) for _ in range(3)]
            if len(set(offsets)) > 1:
                break
        self.offsets = offsets
        indices = list(range(self.MIN_N, self.MAX_N + 1))
        self.rng.shuffle(indices)
        self.held_out = sorted(indices[:3])
        self.queried = set()
        self.predicted = {}
        self.steps = 0
        self.done = False
        obs = (
            "FOUNDRY LEDGER. Every plate n from {} to {} carries a hidden value a(n) "
            "produced by a two-stage rule: an unknown straight-line trend across n, then "
            "an unknown small adjustment that depends only on n's position in a repeating "
            "3-plate cycle. You must determine a(n) for exactly these three RESERVED "
            "plates: {} -- they cannot be queried directly. "
            "Actions (one per step, budget {} steps total): "
            "'QUERY <n>' reveals a(n) for any non-reserved plate n in [{},{}]. "
            "'PREDICT <n> <v>' submits your guess v for a reserved plate n; you learn only "
            "whether it was correct, not the true value. The episode ends once all three "
            "reserved plates have been predicted, or the step budget runs out."
        ).format(
            self.MIN_N, self.MAX_N, self.held_out, self.MAX_STEPS, self.MIN_N, self.MAX_N
        )
        info = {"held_out": list(self.held_out)}
        return obs, info

    def _check_end(self):
        if len(self.predicted) >= 3:
            self.done = True
            return True, False
        if self.steps >= self.MAX_STEPS:
            self.done = True
            return False, True
        return False, False

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        text = (action or "").strip()
        match = re.match(r'^(QUERY|PREDICT)\s+(-?\d+)(?:\s+(-?\d+))?\s*$', text, re.IGNORECASE)

        if not match:
            obs = "Malformed action. Use 'QUERY <n>' or 'PREDICT <n> <v>'."
            terminated, truncated = self._check_end()
            return obs, 0.0, terminated, truncated, {}

        verb = match.group(1).upper()
        n = int(match.group(2))
        extra = match.group(3)

        if verb == "QUERY":
            if extra is not None:
                obs = "QUERY takes exactly one number: 'QUERY <n>'."
                terminated, truncated = self._check_end()
                return obs, 0.0, terminated, truncated, {}
            if n < self.MIN_N or n > self.MAX_N:
                obs = "Plate {} is out of range [{}, {}].".format(n, self.MIN_N, self.MAX_N)
                terminated, truncated = self._check_end()
                return obs, 0.0, terminated, truncated, {}
            if n in self.held_out:
                obs = "Plate {} is reserved for prediction, not query.".format(n)
                terminated, truncated = self._check_end()
                return obs, 0.0, terminated, truncated, {}
            val = self._value(n)
            self.queried.add(n)
            obs = "a({}) = {}".format(n, val)
            terminated, truncated = self._check_end()
            return obs, 0.0, terminated, truncated, {}

        # verb == "PREDICT"
        if extra is None:
            obs = "PREDICT requires a value: 'PREDICT <n> <v>'."
            terminated, truncated = self._check_end()
            return obs, 0.0, terminated, truncated, {}
        v = int(extra)
        if n not in self.held_out:
            obs = "Plate {} is not a reserved prediction target.".format(n)
            terminated, truncated = self._check_end()
            return obs, 0.0, terminated, truncated, {}
        if n in self.predicted:
            obs = "Plate {} was already predicted.".format(n)
            terminated, truncated = self._check_end()
            return obs, 0.0, terminated, truncated, {}

        correct = (v == self._value(n))
        self.predicted[n] = correct
        reward = (1.0 / 3.0) if correct else 0.0
        obs = "Prediction for plate {} was {}.".format(n, "correct" if correct else "incorrect")
        terminated, truncated = self._check_end()
        return obs, reward, terminated, truncated, {}
