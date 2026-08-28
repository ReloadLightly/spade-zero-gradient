import re
import random


class DigitRootChainEnv:
    MAX_STEPS = 10
    REVEALABLE = (1, 2, 3, 4, 5)
    TARGET_INDEX = 6

    def __init__(self):
        self.rng = None
        self.M = None
        self.S = None
        self.terms = []
        self.roots = []
        self.steps = 0
        self.milestone_claimed = False
        self.finished = False

    @staticmethod
    def digit_root(n):
        if n == 0:
            return 0
        r = n % 9
        return 9 if r == 0 else r

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.M = self.rng.choice([2, 3, 4, 5, 6, 7, 8, 9])
        self.S = self.rng.randint(10, 99)
        self.terms = [self.S * (self.M ** i) for i in range(self.TARGET_INDEX + 1)]
        self.roots = [self.digit_root(t) for t in self.terms]
        self.steps = 0
        self.milestone_claimed = False
        self.finished = False

        obs = (
            "DIGIT-ROOT CHAIN. A hidden sequence T_0..T_6 is built by repeatedly "
            "multiplying a hidden 2-digit seed S by a hidden whole-number multiplier M "
            "(2<=M<=9): T_i = S * M^i. You cannot see the numbers themselves, only each "
            "term's digit-root (repeated digit sum until one digit remains). "
            f"T_0's digit-root is {self.roots[0]}.\n"
            "GOAL: determine the digit-root of the far-off term T_6.\n"
            "ACTIONS (send exactly one per turn):\n"
            "  REVEAL <i>           - i in {1,2,3,4,5}; learn the digit-root of T_i\n"
            "  GUESS_MULT_ROOT <d>  - d in 2..9; guess the multiplier M (milestone, partial credit)\n"
            "  GUESS_ROOT <d>       - d in 1..9; guess the digit-root of T_6 (ends episode if correct)\n"
            f"You have {self.MAX_STEPS} steps total."
        )
        return obs, {"digit_root_T0": self.roots[0]}

    def _bounded(self, obs, reward):
        if self.steps >= self.MAX_STEPS:
            self.finished = True
            obs = obs + f" Step limit reached ({self.MAX_STEPS}). Episode over."
            return obs, reward, False, True, {}
        return obs, reward, False, False, {}

    def step(self, action):
        if self.finished:
            return "Episode already finished.", 0.0, True, False, {}
        self.steps += 1
        text = (action or "").strip().upper()

        m = re.match(r"^REVEAL\s+(\d+)$", text)
        if m:
            i = int(m.group(1))
            if i not in self.REVEALABLE:
                return self._bounded(
                    "Invalid index. REVEAL only accepts i in {1,2,3,4,5}.", 0.0
                )
            obs = f"T_{i}'s digit-root is {self.roots[i]}."
            return self._bounded(obs, 0.0)

        m = re.match(r"^GUESS_MULT_ROOT\s+(\d+)$", text)
        if m:
            d = int(m.group(1))
            if not (2 <= d <= 9):
                return self._bounded(
                    "Invalid guess. GUESS_MULT_ROOT needs d in 2..9.", 0.0
                )
            if d == self.M:
                if self.milestone_claimed:
                    return self._bounded(
                        "Correct, but you already claimed this milestone.", 0.0
                    )
                self.milestone_claimed = True
                return self._bounded(f"Correct! The multiplier is {d}.", 0.4)
            direction = "higher" if d < self.M else "lower"
            obs = f"Incorrect. The true multiplier is {direction} than {d}."
            return self._bounded(obs, 0.0)

        m = re.match(r"^GUESS_ROOT\s+(\d+)$", text)
        if m:
            d = int(m.group(1))
            if not (1 <= d <= 9):
                return self._bounded(
                    "Invalid guess. GUESS_ROOT needs d in 1..9.", 0.0
                )
            actual = self.roots[self.TARGET_INDEX]
            if d == actual:
                self.finished = True
                obs = f"Correct! T_6's digit-root is {d}. Episode solved."
                return obs, 0.6, True, False, {}
            direction = "higher" if d < actual else "lower"
            obs = f"Incorrect. The true digit-root is {direction} than {d}."
            return self._bounded(obs, 0.0)

        obs = "Malformed action. Use 'REVEAL <i>', 'GUESS_MULT_ROOT <d>', or 'GUESS_ROOT <d>'."
        return self._bounded(obs, 0.0)
