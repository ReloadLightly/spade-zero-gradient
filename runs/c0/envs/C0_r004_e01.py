import random
import re


class TwoStageSequenceEnv:
    """Two-stage hidden integer sequence: a_n = base(n) + offset[n % m]."""

    TARGETS = (5, 7, 9)
    EXTRA_REVEALABLE = (4, 6, 8)
    INITIAL_REVEAL = (0, 1, 2, 3)
    REWARD_PER_TARGET = {5: 0.34, 7: 0.33, 9: 0.33}
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.base_type = None
        self.mod = None
        self.offsets = None
        self.p = self.q = self.s = 0
        self.known = {}
        self.target_status = {}
        self.step_count = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.base_type = self.rng.choice(["linear", "quadratic", "geometric"])
        if self.base_type == "linear":
            self.p = self.rng.randint(-5, 5)
            self.q = self.rng.randint(1, 4)
        elif self.base_type == "quadratic":
            self.p = self.rng.randint(-3, 3)
            self.q = self.rng.randint(1, 3)
            self.s = self.rng.randint(1, 2)
        else:
            self.p = self.rng.randint(1, 3)
            self.q = self.rng.choice([2, 3])

        self.mod = self.rng.choice([2, 3])
        pool = [x for x in range(-6, 7) if x != 0]
        if self.mod == 2:
            k = self.rng.choice(pool)
            self.offsets = [0, k]
        else:
            k1 = self.rng.choice(pool)
            k2 = self.rng.choice([x for x in pool if x != k1])
            self.offsets = [0, k1, k2]

        self.known = {n: self._value(n) for n in self.INITIAL_REVEAL}
        self.target_status = {n: None for n in self.TARGETS}
        self.step_count = 0
        self.done = False

        known_str = ", ".join(f"a_{n}={v}" for n, v in sorted(self.known.items()))
        obs = (
            "Hidden integer sequence a_n follows a two-stage rule: a smooth base "
            "trend, then a constant offset added per position depending on n's "
            "residue class (n mod m). You do not know the base type, m, or the "
            "offsets.\n"
            f"Known terms: {known_str}\n"
            f"Your goal: predict a_{self.TARGETS[0]}, a_{self.TARGETS[1]}, "
            f"a_{self.TARGETS[2]} correctly.\n"
            f"You may reveal extra evidence terms at indices {self.EXTRA_REVEALABLE} "
            "(indices 5, 7, 9 are reserved for prediction and cannot be revealed).\n"
            "Actions:\n"
            "  REVEAL <n>          -- see a_n for n in the extra-revealable set\n"
            "  PREDICT <n> <value> -- submit your predicted value for target index n\n"
            f"You have {self.MAX_STEPS} steps total. Each target may be predicted once."
        )
        return obs, {"base_type_hidden": True}

    def _base(self, n):
        if self.base_type == "linear":
            return self.p + self.q * n
        if self.base_type == "quadratic":
            return self.p + self.q * n + self.s * n * n
        return self.p * (self.q ** n)

    def _value(self, n):
        return self._base(n) + self.offsets[n % self.mod]

    def _remaining_targets(self):
        return [n for n, s in self.target_status.items() if s is None]

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        reward = 0.0
        text = (action or "").strip()

        m_reveal = re.fullmatch(r"(?i)REVEAL\s+(-?\d+)", text)
        m_predict = re.fullmatch(r"(?i)PREDICT\s+(-?\d+)\s+(-?\d+)", text)

        if m_reveal:
            n = int(m_reveal.group(1))
            if n not in self.EXTRA_REVEALABLE or n in self.known:
                obs = (
                    f"Cannot reveal index {n}. Revealable indices are "
                    f"{[i for i in self.EXTRA_REVEALABLE if i not in self.known]}."
                )
            else:
                self.known[n] = self._value(n)
                obs = (
                    f"a_{n} = {self.known[n]}. "
                    f"Known terms: {sorted(self.known.items())}."
                )
        elif m_predict:
            n = int(m_predict.group(1))
            value = int(m_predict.group(2))
            if n not in self.TARGETS:
                obs = f"Index {n} is not a prediction target. Targets: {self.TARGETS}."
            elif self.target_status[n] is not None:
                obs = f"Target a_{n} already resolved. Remaining: {self._remaining_targets()}."
            else:
                correct = value == self._value(n)
                self.target_status[n] = correct
                if correct:
                    reward = self.REWARD_PER_TARGET[n]
                    obs = f"Correct! a_{n} confirmed. Remaining targets: {self._remaining_targets()}."
                else:
                    obs = f"Incorrect for a_{n}. Remaining targets: {self._remaining_targets()}."
        else:
            obs = (
                "Malformed action. Use 'REVEAL <n>' or 'PREDICT <n> <value>' "
                "with integer arguments."
            )

        all_resolved = all(s is not None for s in self.target_status.values())
        terminated = all_resolved
        truncated = (not terminated) and self.step_count >= self.MAX_STEPS
        if terminated or truncated:
            self.done = True

        info = {"targets_resolved": {n: s for n, s in self.target_status.items()}}
        return obs, reward, terminated, truncated, info
