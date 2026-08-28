import random
import re


class PasswordRiddleEnv:
    """Deduce a hidden 3-digit password (distinct digits 1-5) via bulls/cows feedback."""

    DIGIT_POOL = (1, 2, 3, 4, 5)
    CODE_LEN = 3
    MAX_STEPS = 10
    MILESTONE_REWARDS = {1: 0.2, 2: 0.3, 3: 0.5}

    def __init__(self):
        self.rng = None
        self.secret = None
        self.secret_str = ""
        self.step_count = 0
        self.best_bulls = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.secret = self.rng.sample(self.DIGIT_POOL, self.CODE_LEN)
        self.secret_str = "".join(str(d) for d in self.secret)
        self.step_count = 0
        self.best_bulls = 0
        self.done = False

        parity = "even" if sum(self.secret) % 2 == 0 else "odd"
        obs = (
            "Two riddle-tellers argued over a guarded door, and you overheard "
            "just enough to know the password is a 3-digit code using distinct "
            "digits from 1-5 (e.g. 132). The gatekeeper won't show the password, "
            "but for each guess it will whisper two counts: EXACT (digits that "
            "match both value and position) and DISPLACED (digits that are in "
            "the password but at the wrong position). One riddle-teller let "
            f"slip that the three digits sum to an {parity} number.\n"
            "Action format: 'GUESS' followed by three distinct digits from "
            "1-5, e.g. 'GUESS 214'.\n"
            f"You have {self.MAX_STEPS} steps total. Reach EXACT=3 to win."
        )
        info = {"code_length": self.CODE_LEN, "digit_pool": self.DIGIT_POOL}
        return obs, info

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        match = re.match(r"^\s*GUESS\s+([1-5]{3})\s*$", action.strip(), re.IGNORECASE)

        if not match or len(set(match.group(1))) != self.CODE_LEN:
            obs = (
                "Malformed guess. Use 'GUESS' followed by exactly three "
                "distinct digits from 1-5, e.g. 'GUESS 253'."
            )
            reward = 0.0
            terminated = False
            truncated = self.step_count >= self.MAX_STEPS
            if truncated:
                self.done = True
                obs += f" No guesses remain. The password was {self.secret_str}."
            info = {"valid": False, "steps_remaining": max(0, self.MAX_STEPS - self.step_count)}
            return obs, reward, terminated, truncated, info

        guess = [int(c) for c in match.group(1)]
        bulls = sum(1 for g, s in zip(guess, self.secret) if g == s)
        cows = sum(1 for d in guess if d in self.secret) - bulls

        reward = 0.0
        if bulls > self.best_bulls:
            for level in range(self.best_bulls + 1, bulls + 1):
                reward += self.MILESTONE_REWARDS.get(level, 0.0)
            self.best_bulls = bulls

        terminated = bulls == self.CODE_LEN
        truncated = (not terminated) and self.step_count >= self.MAX_STEPS
        if terminated or truncated:
            self.done = True

        if terminated:
            obs = (
                f"EXACT={bulls}, DISPLACED={cows}. The door swings open — "
                f"{self.secret_str} was the password!"
            )
        elif truncated:
            obs = (
                f"EXACT={bulls}, DISPLACED={cows}. Out of attempts — "
                f"the password was {self.secret_str}."
            )
        else:
            obs = (
                f"EXACT={bulls}, DISPLACED={cows}. The gatekeeper waits for "
                f"your next guess ({self.MAX_STEPS - self.step_count} left)."
            )

        info = {
            "valid": True,
            "bulls": bulls,
            "cows": cows,
            "best_bulls": self.best_bulls,
            "steps_remaining": max(0, self.MAX_STEPS - self.step_count),
        }
        return obs, reward, terminated, truncated, info
