import random
import re


class TwoStageSequenceEnv:
    """Hidden sequence a(n) = m*n + b, doubled whenever n % k == 0.

    The solver must query sample indices to infer m, b, and k, optionally
    report m and k for partial credit, then predict a(target) for an
    index it is not permitted to query directly.
    """

    MAX_STEPS = 10
    INDEX_LO, INDEX_HI = 1, 30

    _QUERY_RE = re.compile(r"^\s*QUERY\s+(-?\d+)\s*$", re.IGNORECASE)
    _REPORT_M_RE = re.compile(r"^\s*REPORT\s+m\s*=\s*(-?\d+)\s*$", re.IGNORECASE)
    _REPORT_K_RE = re.compile(r"^\s*REPORT\s+k\s*=\s*(-?\d+)\s*$", re.IGNORECASE)
    _PREDICT_RE = re.compile(r"^\s*PREDICT\s+(-?\d+)\s*$", re.IGNORECASE)

    def __init__(self):
        self.rng = None
        self.m = None
        self.b = None
        self.k = None
        self.target = None
        self.step_count = 0
        self.queried = set()
        self.m_awarded = False
        self.k_awarded = False
        self.done = False

    def _value(self, n: int) -> int:
        base = self.m * n + self.b
        if n % self.k == 0:
            return base * 2
        return base

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.m = self.rng.choice([2, 3, 4])
        self.b = self.rng.choice([1, 2, 3, 4, 5])
        self.k = self.rng.choice([3, 4, 5])
        self.target = self.rng.randint(22, self.INDEX_HI)

        self.step_count = 0
        self.queried = set()
        self.m_awarded = False
        self.k_awarded = False
        self.done = False

        obs = (
            "SEQUENCE PROBE. A hidden sequence follows a(n) = m*n + b for "
            "integer n, EXCEPT that whenever n is a multiple of a hidden "
            f"modulus k, the value is doubled. Indices run {self.INDEX_LO}"
            f"-{self.INDEX_HI}. Your goal: determine a({self.target}) "
            f"without ever querying index {self.target} directly.\n"
            "Actions (exactly one per turn):\n"
            "  QUERY n        -- reveal a(n) for any index except the target\n"
            "  REPORT m=X     -- submit your guess for the hidden slope m "
            "(one shot at reward, retries give feedback only)\n"
            "  REPORT k=X     -- submit your guess for the hidden modulus k\n"
            f"  PREDICT value  -- submit your final guess for a({self.target}); "
            "this ends the episode\n"
            f"You have {self.MAX_STEPS} steps total (every action, valid or "
            "not, counts as one step). Correct REPORT m and REPORT k each "
            "earn partial credit; a correct PREDICT earns the rest."
        )
        return obs, {"target": self.target}

    def step(self, action: str):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        reward = 0.0
        terminated = False

        m = self._QUERY_RE.match(action or "")
        if m:
            n = int(m.group(1))
            if n == self.target:
                obs = (
                    f"Index {self.target} is the target and cannot be "
                    "queried directly. Choose a different index."
                )
            elif not (self.INDEX_LO <= n <= self.INDEX_HI):
                obs = (
                    f"Index {n} is out of range "
                    f"[{self.INDEX_LO}, {self.INDEX_HI}]."
                )
            else:
                self.queried.add(n)
                obs = f"a({n}) = {self._value(n)}"
            return self._finish_step(obs, reward, terminated)

        m = self._REPORT_M_RE.match(action or "")
        if m:
            guess = int(m.group(1))
            if guess == self.m:
                if not self.m_awarded:
                    reward = 0.3
                    self.m_awarded = True
                    obs = "REPORT m correct. Partial credit awarded."
                else:
                    obs = "REPORT m correct (already credited)."
            else:
                direction = "too low" if guess < self.m else "too high"
                obs = f"REPORT m incorrect: {direction}."
            return self._finish_step(obs, reward, terminated)

        m = self._REPORT_K_RE.match(action or "")
        if m:
            guess = int(m.group(1))
            if guess == self.k:
                if not self.k_awarded:
                    reward = 0.3
                    self.k_awarded = True
                    obs = "REPORT k correct. Partial credit awarded."
                else:
                    obs = "REPORT k correct (already credited)."
            else:
                direction = "too low" if guess < self.k else "too high"
                obs = f"REPORT k incorrect: {direction}."
            return self._finish_step(obs, reward, terminated)

        m = self._PREDICT_RE.match(action or "")
        if m:
            guess = int(m.group(1))
            correct = self._value(self.target)
            terminated = True
            self.done = True
            if guess == correct:
                reward = 0.4
                obs = (
                    f"PREDICT correct: a({self.target}) = {correct}. "
                    "Episode complete."
                )
            else:
                obs = (
                    f"PREDICT incorrect: you guessed {guess}, true value "
                    f"was {correct}. Episode complete."
                )
            return self._finish_step(obs, reward, terminated)

        obs = (
            "Malformed action. Use one of: 'QUERY n', 'REPORT m=X', "
            "'REPORT k=X', or 'PREDICT value'."
        )
        return self._finish_step(obs, reward, terminated)

    def _finish_step(self, obs, reward, terminated):
        truncated = False
        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs = obs + " Step limit reached; episode truncated."
        if terminated:
            self.done = True
        info = {
            "step": self.step_count,
            "m_awarded": self.m_awarded,
            "k_awarded": self.k_awarded,
        }
        return obs, reward, terminated, truncated, info
