"""Deterministic fixtures for tests and machinery smoke (no model calls)."""

GOOD_ENV = '''import random

class SecretDigitEnv:
    """Guess a secret digit 0-9. Feedback: too low / too high. 6 steps."""

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.target = self.rng.randint(0, 9)
        self.steps_left = 6
        obs = ("A secret digit between 0 and 9 has been chosen. "
               "Guess it within 6 steps. Action format: a single digit, e.g. 4.")
        return obs, {}

    def step(self, action):
        self.steps_left -= 1
        trunc = self.steps_left <= 0
        try:
            g = int(str(action).strip())
        except ValueError:
            return ("Please guess a single digit 0-9.", 0.0, False, trunc, {})
        if g == self.target:
            return ("Correct!", 1.0, True, False, {})
        fb = "Too low." if g < self.target else "Too high."
        return (fb + f" {self.steps_left} steps left.", 0.0, False, trunc, {})
'''

BROKEN_ENV = '''import os

class LeakyEnv:
    def reset(self, seed=None):
        return os.getcwd(), {}

    def step(self, action):
        return "", 0.0, True, False, {}
'''

NONDET_ENV = '''import random

class DriftyEnv:
    def reset(self, seed=None):
        self.target = random.randint(0, 9)   # unseeded module RNG — contract violation
        self.steps_left = 3
        return "Guess.", {}

    def step(self, action):
        self.steps_left -= 1
        return str(self.target), 0.0, False, self.steps_left <= 0, {}
'''


def _parse_history(prompt: str):
    """Extract (guess, feedback) pairs from the runner's transcript format."""
    guesses, feedbacks = [], []
    for block in prompt.split("[YOU]")[1:]:
        reply = block.split("[OBSERVATION]")[0]
        for line in reply.splitlines():
            if line.startswith("ACTION:"):
                guesses.append(line.split("ACTION:", 1)[1].strip())
        if "[OBSERVATION]" in block:
            feedbacks.append(block.split("[OBSERVATION]", 1)[1].split("[")[0])
    return guesses, feedbacks


def solver_policy(prompt: str, model: str) -> str:
    """Mock solver for SecretDigitEnv.

    With HINT: binary search (always wins within 6 steps).
    Without: sequential 0,1,2,... (wins only for small targets) -> regret > 0.
    """
    guesses, feedbacks = _parse_history(prompt)
    if "HINT:" in prompt:
        lo, hi = 0, 9
        for g, fb in zip(guesses, feedbacks):
            try:
                gi = int(g)
            except ValueError:
                continue
            if "Too low" in fb:
                lo = max(lo, gi + 1)
            elif "Too high" in fb:
                hi = min(hi, gi - 1)
        nxt = (lo + hi) // 2
    else:
        nxt = len(guesses)
    return f"Thinking about it.\nACTION: {nxt}"
