import random
import re


class TwoStageResidueEnv:
    def __init__(self):
        self.N = 30
        self.max_steps = 10
        self.rng = None
        self.m = None
        self.a = None
        self.b = None
        self.queried = set()
        self.predicted_correct = set()
        self.correct_count = 0
        self.total_reward = 0.0
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.m = self.rng.choice([2, 2, 3])
        while True:
            a = [self.rng.randint(1, 5) for _ in range(self.m)]
            b = [self.rng.randint(-4, 4) for _ in range(self.m)]
            if len(set(zip(a, b))) > 1:
                break
        self.a, self.b = a, b
        self.queried = set()
        self.predicted_correct = set()
        self.correct_count = 0
        self.total_reward = 0.0
        self.steps = 0
        self.done = False

        obs = (
            "GOAL: A hidden rule generates an integer sequence over positions "
            "1..{n}. The rule works in two stages: each position is first "
            "sorted into one of a small, fixed number of classes by its "
            "position, then a class-specific linear formula (its own slope "
            "and offset) produces the value.\n"
            "ACTIONS (exactly one per turn):\n"
            "  QUERY <n>            reveal the sequence value at position n\n"
            "  PREDICT <n> <value>  claim the value at a position you have "
            "NOT queried\n"
            "You have {steps} total actions (queries and predictions "
            "combined). Reward is earned only for PREDICT calls on positions "
            "you never queried: your first 3 distinct correct out-of-sample "
            "predictions complete the episode for full reward (partial "
            "credit per correct one). Predicting an already-queried "
            "position earns nothing.".format(n=self.N, steps=self.max_steps)
        )
        info = {"n_range": self.N, "max_steps": self.max_steps}
        return obs, info

    def _value(self, n):
        c = n % self.m
        return self.a[c] * n + self.b[c]

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        reward = 0.0
        terminated = False
        info = {
            "steps_used": self.steps,
            "correct_count": self.correct_count,
            "queried_count": len(self.queried),
        }

        m = re.match(
            r"^\s*(QUERY|PREDICT)\s+(-?\d+)(?:\s+(-?\d+))?\s*$",
            action.strip(),
            re.IGNORECASE,
        )
        if not m:
            obs = (
                "Malformed action. Use 'QUERY <n>' or 'PREDICT <n> <value>' "
                "with n between 1 and {}.".format(self.N)
            )
        else:
            verb = m.group(1).upper()
            n = int(m.group(2))
            if not (1 <= n <= self.N):
                obs = "Position out of range: n must satisfy 1 <= n <= {}.".format(
                    self.N
                )
            elif verb == "QUERY":
                val = self._value(n)
                self.queried.add(n)
                obs = "Value at position {} is {}.".format(n, val)
            else:
                if m.group(3) is None:
                    obs = "PREDICT requires a value: 'PREDICT <n> <value>'."
                else:
                    guess = int(m.group(3))
                    true_val = self._value(n)
                    if n in self.queried:
                        obs = (
                            "Position {} was already revealed by a query; a "
                            "prediction there earns no reward (it doesn't "
                            "demonstrate generalization).".format(n)
                        )
                    elif n in self.predicted_correct:
                        obs = "Position {} was already credited.".format(n)
                    elif guess == true_val:
                        self.predicted_correct.add(n)
                        self.correct_count += 1
                        if self.correct_count < 3:
                            reward = 1.0 / 3.0
                        else:
                            reward = 1.0 - self.total_reward
                        self.total_reward += reward
                        obs = (
                            "Correct out-of-sample prediction at position {} "
                            "({}/3 needed).".format(n, self.correct_count)
                        )
                        if self.correct_count >= 3:
                            terminated = True
                            self.done = True
                    else:
                        direction = "too high" if guess > true_val else "too low"
                        obs = (
                            "Incorrect prediction at position {}: your guess "
                            "was {}.".format(n, direction)
                        )

        truncated = False
        if not terminated and self.steps >= self.max_steps:
            truncated = True
            self.done = True

        info["correct_count"] = self.correct_count
        info["steps_used"] = self.steps
        return obs, reward, terminated, truncated, info
