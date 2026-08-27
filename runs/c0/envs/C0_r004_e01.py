import random
import re


class TwoStageSequenceEnv:
    """Hidden two-stage integer sequence: linear base + periodic modifier."""

    MAX_N = 30
    STEP_LIMIT = 10

    QUERY_RE = re.compile(r'^\s*QUERY\s+(-?\d+)\s*$', re.IGNORECASE)
    GUESS_RE = re.compile(
        r'^\s*GUESS\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$',
        re.IGNORECASE,
    )

    def __init__(self):
        self.rng = None
        self.p = None
        self.q = None
        self.k = None
        self.r = None
        self.d = None
        self.targets = None
        self.steps = 0
        self.done = False

    def _value(self, n):
        base = self.p * n + self.q
        if n % self.k == self.r:
            base += self.d
        return base

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.p = self.rng.randint(2, 4)
        self.q = self.rng.randint(0, 5)
        self.k = self.rng.choice([2, 3])
        self.r = self.rng.randint(0, self.k - 1)
        magnitude = self.rng.randint(3, 9)
        self.d = magnitude * self.rng.choice([1, -1])

        pool = list(range(6, self.MAX_N + 1))
        self.rng.shuffle(pool)
        self.targets = sorted(pool[:3])
        self.rng.shuffle(self.targets)

        self.steps = 0
        self.done = False

        obs = (
            "Hidden sequence a(n) is defined for integer n in [0, 30] by a fixed "
            "but unknown two-stage rule (a linear pattern in n, modified at a "
            "periodic subset of indices). You have {steps} steps total.\n"
            "Actions:\n"
            "  QUERY <n>            reveal a(n) for any n in [0, 30] (costs a step, no reward)\n"
            "  GUESS <n1> <v1> <n2> <v2> <n3> <v3>   submit your predicted values for "
            "EXACTLY these three target indices (any order), ending the episode\n"
            "Target indices to predict: {targets}\n"
            "Reward: 0.33 for each of the first two correct target values, 0.34 for "
            "the third (max total 1.0). Figure out the rule by querying other "
            "indices before you guess.".format(steps=self.STEP_LIMIT, targets=self.targets)
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False

        qm = self.QUERY_RE.match(action or "")
        gm = self.GUESS_RE.match(action or "")

        if qm:
            n = int(qm.group(1))
            if 0 <= n <= self.MAX_N:
                obs = "a({}) = {}".format(n, self._value(n))
            else:
                obs = "Invalid index {}. n must be in [0, 30]. Try QUERY <n> again.".format(n)
        elif gm:
            nums = [int(x) for x in gm.groups()]
            pairs = {}
            valid_indices = True
            for i in range(0, 6, 2):
                idx, val = nums[i], nums[i + 1]
                pairs[idx] = val
            if sorted(pairs.keys()) != sorted(self.targets):
                valid_indices = False

            if not valid_indices:
                obs = (
                    "GUESS must give a value for exactly the three announced target "
                    "indices {}, each once. Try again.".format(self.targets)
                )
            else:
                weights = [0.33, 0.33, 0.34]
                correct_flags = []
                for i, t in enumerate(self.targets):
                    actual = self._value(t)
                    if pairs[t] == actual:
                        reward += weights[i]
                        correct_flags.append(True)
                    else:
                        correct_flags.append(False)
                reward = round(reward, 2)
                terminated = True
                self.done = True
                obs = "GUESS scored. Correct: {}. Reward earned: {:.2f}.".format(
                    dict(zip(self.targets, correct_flags)), reward
                )
        else:
            obs = (
                "Malformed action. Use 'QUERY <n>' or "
                "'GUESS <n1> <v1> <n2> <v2> <n3> <v3>' with the three target indices."
            )

        if not terminated and self.steps >= self.STEP_LIMIT:
            truncated = True
            self.done = True
            obs += " Step limit reached; episode over."

        return obs, reward, terminated, truncated, {}
