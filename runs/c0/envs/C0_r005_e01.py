import random
import re


class TwoStageSequenceEnv:
    def __init__(self):
        self.rng = None
        self.k = 0
        self.A = []
        self.B = []
        self.N = 18
        self.f = []
        self.targets = []
        self.queried = {}
        self.correct_targets = set()
        self.step_count = 0
        self.max_steps = 10
        self.reward_schedule = [0.34, 0.33, 0.33]
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.k = self.rng.choice([2, 3])
        self.A = [self.rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(self.k)]
        self.B = [self.rng.randint(-6, 6) for _ in range(self.k)]
        self.N = 18
        self.f = []
        for i in range(self.N):
            c = i % self.k
            q = i // self.k
            self.f.append(self.A[c] * q + self.B[c])
        upper_pool = list(range(self.N - 6, self.N))
        self.rng.shuffle(upper_pool)
        self.targets = sorted(upper_pool[:3])
        self.queried = {}
        self.correct_targets = set()
        self.step_count = 0
        self.done = False
        obs = (
            "SEQUENCE PUZZLE: a hidden rule assigns an integer f(i) to every index i in "
            "0..{n1}. The rule works in two stages: each index secretly belongs to one of a "
            "few classes, and within a class the values follow their own straight-line "
            "pattern as you move along that class. GOAL: state the correct value of f(i) "
            "for each LOCKED target index in {t}; locked indices cannot be queried directly. "
            "ACTIONS (send exactly one per turn):\n"
            "  QUERY <i>       reveal f(i) for any unlocked index i in 0..{n1}\n"
            "  PREDICT <i> <v> claim f(i) = v for a locked target index i\n"
            "You have {steps} steps total. Correctly predicting all locked targets ends the "
            "episode with full reward."
        ).format(n1=self.N - 1, t=self.targets, steps=self.max_steps)
        return obs, {"targets": list(self.targets), "n": self.N}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        reward = 0.0
        terminated = False

        text = (action or "").strip()
        qm = re.match(r'^QUERY\s+(-?\d+)$', text, re.IGNORECASE)
        pm = re.match(r'^PREDICT\s+(-?\d+)\s+(-?\d+)$', text, re.IGNORECASE)

        if qm:
            i = int(qm.group(1))
            if i < 0 or i >= self.N:
                obs = f"Invalid index {i}: must be in 0..{self.N - 1}. No step effect beyond the count."
            elif i in self.targets:
                obs = f"Index {i} is a LOCKED target; it cannot be queried. Use PREDICT {i} <v> instead."
            else:
                value = self.f[i]
                self.queried[i] = value
                known = ", ".join(f"f({j})={v}" for j, v in sorted(self.queried.items()))
                obs = f"f({i}) = {value}. Known values so far: {known}."
        elif pm:
            i = int(pm.group(1))
            v = int(pm.group(2))
            if i not in self.targets:
                obs = f"Index {i} is not a locked target ({self.targets}); nothing to predict there."
            elif i in self.correct_targets:
                obs = f"Target {i} is already solved."
            else:
                true_val = self.f[i]
                if v == true_val:
                    self.correct_targets.add(i)
                    reward = self.reward_schedule[len(self.correct_targets) - 1]
                    remaining = [t for t in self.targets if t not in self.correct_targets]
                    if remaining:
                        obs = f"Correct: f({i}) = {v}. Remaining locked targets: {remaining}."
                    else:
                        obs = f"Correct: f({i}) = {v}. All targets solved!"
                        terminated = True
                        self.done = True
                else:
                    direction = "higher" if v < true_val else "lower"
                    obs = f"Incorrect: the true value of f({i}) is {direction} than {v}."
        else:
            obs = "Malformed action. Use 'QUERY <i>' or 'PREDICT <i> <v>' with integer arguments."

        truncated = False
        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            self.done = True
            unsolved = [t for t in self.targets if t not in self.correct_targets]
            if unsolved:
                reveal = ", ".join(f"f({t})={self.f[t]}" for t in unsolved)
                obs += f" Step limit reached. Unsolved targets revealed: {reveal}."

        info = {"queried": dict(self.queried), "solved": sorted(self.correct_targets), "step": self.step_count}
        return obs, reward, terminated, truncated, info
