import random
import re


class PopulationRegimeEnv:
    def __init__(self):
        self.rng = None
        self.p0 = None
        self.d = None
        self.m = None
        self.k = None
        self.observable_max = 10
        self.classify_year = None
        self.target_year = None
        self.sequence = None
        self.steps = 0
        self.max_steps = 10
        self.classify_used = False
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.p0 = self.rng.randint(20, 60)
        self.d = self.rng.randint(3, 10)
        self.m = self.rng.choice([2, 3])
        self.k = self.rng.randint(2, 3)
        self.observable_max = 10
        self.classify_year = self.rng.randint(3, 8)
        self.target_year = self.rng.randint(11, 13)
        self.sequence = self._simulate(self.target_year + 1)
        self.steps = 0
        self.max_steps = 10
        self.classify_used = False
        self.done = False
        obs = (
            "CENSUS RECONSTRUCTION\n"
            "A settlement's population follows a hidden yearly sequence starting at year 0. "
            "It alternates between two regimes: STEADY (population increases by a fixed amount "
            "each year) and BOOM (population multiplies by a fixed factor each year). The active "
            "regime switches every fixed number of years (2 or 3), STEADY governing year 0.\n"
            f"You may query known years with 'OBSERVE <year>' for any year from 0 to "
            f"{self.observable_max} (each query costs one step).\n"
            f"Once you believe you know the regime governing the transition from year "
            f"{self.classify_year} to year {self.classify_year + 1}, submit "
            "'CLASSIFY <STEADY|BOOM>' (scored once, worth 0.3).\n"
            f"Finally, submit 'PREDICT <value>' for the population at year {self.target_year} "
            "(beyond the observable range) — worth 0.7 if exact, and ends the episode.\n"
            f"You have at most {self.max_steps} steps total. Malformed actions cost a step and "
            "earn no reward."
        )
        info = {"target_year": self.target_year, "classify_year": self.classify_year}
        return obs, info

    def _simulate(self, n):
        seq = [self.p0]
        pop = self.p0
        for t in range(n - 1):
            regime = self._regime_at(t)
            if regime == 'STEADY':
                pop = pop + self.d
            else:
                pop = pop * self.m
            seq.append(pop)
        return seq

    def _regime_at(self, t):
        block = t // self.k
        return 'STEADY' if block % 2 == 0 else 'BOOM'

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}
        self.steps += 1
        text = (action or "").strip()

        m_obs = re.match(r'(?i)^OBSERVE\s+(-?\d+)$', text)
        m_cls = re.match(r'(?i)^CLASSIFY\s+(STEADY|BOOM)$', text)
        m_pred = re.match(r'(?i)^PREDICT\s+(-?\d+)$', text)

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if m_obs:
            year = int(m_obs.group(1))
            if 0 <= year <= self.observable_max:
                obs = f"Year {year}: population = {self.sequence[year]}."
            else:
                obs = f"Invalid year. OBSERVE only accepts years 0 to {self.observable_max}."
        elif m_cls:
            guess = m_cls.group(1).upper()
            if self.classify_used:
                obs = "You have already used your CLASSIFY attempt."
            else:
                self.classify_used = True
                correct = self._regime_at(self.classify_year)
                if guess == correct:
                    reward = 0.3
                    obs = f"Correct: the regime active at year {self.classify_year} is {correct}."
                else:
                    obs = "Incorrect classification. (No further CLASSIFY attempts remain.)"
        elif m_pred:
            guess_val = int(m_pred.group(1))
            actual = self.sequence[self.target_year]
            terminated = True
            self.done = True
            if guess_val == actual:
                reward = 0.7
                obs = f"Correct! Population at year {self.target_year} was {actual}. Episode complete."
            else:
                obs = f"Incorrect. Population at year {self.target_year} was {actual}. Episode complete."
        else:
            obs = ("Malformed action. Use one of: 'OBSERVE <year>', "
                   "'CLASSIFY <STEADY|BOOM>', or 'PREDICT <value>'.")

        if not terminated and self.steps >= self.max_steps:
            truncated = True
            self.done = True
            obs += " Step limit reached; episode over."

        return obs, reward, terminated, truncated, info
