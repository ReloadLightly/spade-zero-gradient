import random
import re


class VaultRiderRiddleEnv:
    MAX_STEPS = 8
    DIGIT_POOL = list(range(1, 10))
    PRIMES = {2, 3, 5, 7}

    def __init__(self):
        self.rng = None
        self.secret = None
        self.steps = 0
        self.best_bulls = 0
        self.clues = []
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.secret = self.rng.sample(self.DIGIT_POOL, 4)
        self.steps = 0
        self.best_bulls = 0
        self.done = False
        self.clues = self._make_clues()
        return self._intro(), {"clues": list(self.clues)}

    def _make_clues(self):
        generators = [
            self._clue_sum, self._clue_parity, self._clue_compare,
            self._clue_prime_count, self._clue_max, self._clue_min,
        ]
        chosen = self.rng.sample(generators, 2)
        return [g() for g in chosen]

    def _clue_sum(self):
        return f"The four digits sum to exactly {sum(self.secret)}."

    def _clue_parity(self):
        i = self.rng.randrange(4)
        parity = "even" if self.secret[i] % 2 == 0 else "odd"
        return f"The digit in position {i + 1} is {parity}."

    def _clue_compare(self):
        i, j = self.rng.sample(range(4), 2)
        if self.secret[i] > self.secret[j]:
            return f"The digit in position {i + 1} is greater than the digit in position {j + 1}."
        return f"The digit in position {i + 1} is less than the digit in position {j + 1}."

    def _clue_prime_count(self):
        k = sum(1 for d in self.secret if d in self.PRIMES)
        return f"Exactly {k} of the four digits are prime numbers (from 2, 3, 5, 7)."

    def _clue_max(self):
        return f"No digit in the password exceeds {max(self.secret)}."

    def _clue_min(self):
        return f"The smallest digit in the password is {min(self.secret)}."

    def _intro(self):
        lines = [
            "You must recover a 4-digit vault password. It uses 4 DISTINCT digits from "
            "1-9 in a fixed order (positions 1-4, left to right).",
            "Two fragments overheard about the password:",
        ]
        for c in self.clues:
            lines.append(f"  - {c}")
        lines.append(
            "On each turn, submit a full guess: 'GUESS d1 d2 d3 d4' "
            "(four distinct digits 1-9, e.g. 'GUESS 3 7 1 9')."
        )
        lines.append(
            "After each guess you learn BULLS (digits correct AND in the correct "
            "position) and COWS (digits that belong in the password but are in the "
            "wrong position)."
        )
        lines.append(
            f"You have {self.MAX_STEPS} guesses total. Reward is earned only when your "
            "BULLS count exceeds your previous best this episode; reaching 4 bulls ends "
            "the episode with full reward."
        )
        return "\n".join(lines)

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}
        self.steps += 1
        parsed = self._parse(action)
        if parsed is None:
            truncated = self.steps >= self.MAX_STEPS
            if truncated:
                self.done = True
            obs = (
                "Malformed action. Use exactly: GUESS d1 d2 d3 d4 (four DISTINCT "
                f"digits from 1-9, space-separated). Guesses used: "
                f"{self.steps}/{self.MAX_STEPS}."
            )
            return obs, 0.0, False, truncated, {"valid": False}

        bulls, cows = self._score(parsed)
        reward = 0.0
        if bulls > self.best_bulls:
            reward = (bulls - self.best_bulls) * 0.25
            self.best_bulls = bulls
        terminated = bulls == 4
        truncated = (not terminated) and self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.done = True
        obs = self._feedback_text(parsed, bulls, cows, terminated, truncated)
        info = {"bulls": bulls, "cows": cows, "best_bulls": self.best_bulls, "steps": self.steps}
        return obs, reward, terminated, truncated, info

    def _parse(self, action):
        if not isinstance(action, str):
            return None
        tokens = action.strip().split()
        if len(tokens) != 5 or tokens[0].upper() != "GUESS":
            return None
        digits = []
        for t in tokens[1:]:
            if not re.fullmatch(r"[1-9]", t):
                return None
            digits.append(int(t))
        if len(set(digits)) != 4:
            return None
        return digits

    def _score(self, guess):
        bulls = sum(1 for i in range(4) if guess[i] == self.secret[i])
        cows = len(set(guess) & set(self.secret)) - bulls
        return bulls, cows

    def _feedback_text(self, guess, bulls, cows, terminated, truncated):
        lines = [
            f"Guess {self.steps}: {' '.join(str(d) for d in guess)} -> "
            f"{bulls} bulls, {cows} cows."
        ]
        if terminated:
            lines.append("Exact match! The vault password is recovered.")
        elif truncated:
            lines.append(f"Out of guesses ({self.MAX_STEPS}/{self.MAX_STEPS} used). The vault remains locked.")
        else:
            remaining = self.MAX_STEPS - self.steps
            lines.append(
                f"Best bulls so far this episode: {self.best_bulls}. "
                f"Guesses remaining: {remaining}."
            )
        return "\n".join(lines)
