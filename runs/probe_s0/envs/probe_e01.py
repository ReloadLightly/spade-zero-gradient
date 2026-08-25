import random
import re
import string


class RLEHiddenRuleEnv:
    """Infer a hidden run-length expansion rule from probes, then predict a target expansion."""

    LETTERS = string.ascii_lowercase[:10]
    STEP_LIMIT = 10
    _ACTION_RE = re.compile(r"^PREDICT\s+(\S+)\s+(\S+)$")

    def __init__(self):
        self.rng = None
        self.order = None
        self.shift = None
        self.target_token = None
        self.target_expansion = None
        self.revealed = {}
        self.novel_hits = 0
        self.steps = 0
        self.done = False

    def _make_run(self):
        count = self.rng.randint(1, 4)
        letter = self.rng.choice(self.LETTERS)
        return count, letter

    def _make_token(self, runs):
        parts = []
        for count, letter in runs:
            if self.order == "digit_first":
                parts.append(f"{count}{letter}")
            else:
                parts.append(f"{letter}{count}")
        return "".join(parts)

    def _expand(self, runs):
        out = []
        for count, letter in runs:
            shifted = chr((ord(letter) - ord('a') + self.shift) % 26 + ord('a'))
            out.append(shifted * count)
        return "".join(out)

    def _random_runs(self):
        n = self.rng.randint(2, 3)
        return [self._make_run() for _ in range(n)]

    def _fresh_token(self, existing):
        for _ in range(50):
            runs = self._random_runs()
            token = self._make_token(runs)
            if token not in existing:
                return token, runs
        runs = self._random_runs()
        return self._make_token(runs), runs

    def _expand_token_string(self, token):
        if self.order == "digit_first":
            pattern = re.compile(r"(\d)([a-j])")
        else:
            pattern = re.compile(r"([a-j])(\d)")
        pos = 0
        runs = []
        while pos < len(token):
            m = pattern.match(token, pos)
            if not m:
                return None
            if self.order == "digit_first":
                count, letter = int(m.group(1)), m.group(2)
            else:
                letter, count = m.group(1), int(m.group(2))
            if count < 1:
                return None
            runs.append((count, letter))
            pos = m.end()
        if not runs:
            return None
        return self._expand(runs)

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.order = self.rng.choice(["digit_first", "letter_first"])
        self.shift = self.rng.randint(0, 4)
        self.steps = 0
        self.done = False
        self.novel_hits = 0
        self.revealed = {}

        examples = []
        for _ in range(2):
            token, runs = self._fresh_token(self.revealed)
            expansion = self._expand(runs)
            self.revealed[token] = expansion
            examples.append((token, expansion))

        target_token, target_runs = self._fresh_token(self.revealed)
        self.target_token = target_token
        self.target_expansion = self._expand(target_runs)
        self.revealed[target_token] = self.target_expansion

        ex_lines = "\n".join(f"  {t} -> {e}" for t, e in examples)
        obs = (
            "Tokens are compressed runs of COUNT+LETTER or LETTER+COUNT (one fixed order "
            "used throughout this episode). Each run means 'repeat a letter COUNT times', "
            "but the letter is first shifted by a fixed hidden amount (a Caesar shift) "
            "before repeating.\n"
            f"Examples of the hidden rule applied:\n{ex_lines}\n"
            f"Your target token to expand is: {self.target_token}\n"
            "Actions (one per step): PREDICT <token> <expansion>\n"
            "  If <token> is new (not shown above and not already predicted), a correct "
            "prediction proves your hypothesis and scores partial credit; a wrong one "
            "reveals the true expansion so you can refine your hypothesis.\n"
            f"  Predicting {self.target_token} correctly wins the episode; wrong guesses "
            "on it are told only right/wrong, with no reveal.\n"
            f"You have {self.STEP_LIMIT} steps total, letters used are 'a'-'j' only."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        match = self._ACTION_RE.match(action.strip())

        if not match:
            obs = "Malformed action. Use: PREDICT <token> <expansion>"
            truncated = self.steps >= self.STEP_LIMIT
            self.done = truncated
            return obs, 0.0, False, truncated, {}

        token, guess = match.group(1), match.group(2)

        if token == self.target_token:
            if guess == self.target_expansion:
                obs = f"Correct! {token} -> {guess}. Target solved."
                self.done = True
                return obs, 0.4, True, False, {}
            obs = "Incorrect for the target token. No reveal. Keep probing other tokens."
            truncated = self.steps >= self.STEP_LIMIT
            self.done = truncated
            return obs, 0.0, False, truncated, {}

        already_seen = token in self.revealed
        actual = self.revealed.get(token)
        if actual is None:
            actual = self._expand_token_string(token)
            if actual is None:
                obs = (
                    f"'{token}' is not a valid token under the fixed order (all tokens use "
                    "the same COUNT/LETTER order and letters 'a'-'j'). Try a differently formed token."
                )
                truncated = self.steps >= self.STEP_LIMIT
                self.done = truncated
                return obs, 0.0, False, truncated, {}
            self.revealed[token] = actual

        correct = guess == actual
        if correct and not already_seen and self.novel_hits < 2:
            self.novel_hits += 1
            obs = f"Correct! {token} -> {guess}. Hypothesis confirmed ({self.novel_hits}/2 credited)."
            reward = 0.3
        elif correct:
            obs = f"Correct! {token} -> {guess}. (No further credit for repeated or extra confirmations.)"
            reward = 0.0
        else:
            obs = f"Incorrect. {token} actually expands to: {actual}"
            reward = 0.0

        truncated = self.steps >= self.STEP_LIMIT
        self.done = truncated
        return obs, reward, False, truncated, {}
