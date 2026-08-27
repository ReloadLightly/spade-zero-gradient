import random
import re
import string


class MysteryBaseEnv:
    def __init__(self):
        self.rng = None
        self.base = None
        self.alphabet = []
        self.symbol_to_digit = {}
        self.target_value = None
        self.target_numeral = ""
        self.steps = 0
        self.max_steps = 10
        self.base_submitted = False
        self.value_submitted = False

    def _encode(self, n):
        if n == 0:
            return self.alphabet[0]
        digits = []
        x = n
        while x > 0:
            digits.append(x % self.base)
            x //= self.base
        digits.reverse()
        return ''.join(self.alphabet[d] for d in digits)

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.base = self.rng.randint(6, 16)
        pool = list(string.ascii_uppercase)
        self.rng.shuffle(pool)
        self.alphabet = pool[:self.base]
        self.symbol_to_digit = {s: i for i, s in enumerate(self.alphabet)}

        low = self.base
        high = self.base * self.base - 1
        self.target_value = self.rng.randint(low, high)
        self.target_numeral = self._encode(self.target_value)

        self.steps = 0
        self.base_submitted = False
        self.value_submitted = False

        obs = (
            "A hidden numeral system encodes non-negative integers in an unknown "
            "base B (6 <= B <= 16), using letters (A-Z) as digit symbols under a "
            "hidden fixed mapping (digit value -> letter), the same for every number.\n"
            f"Challenge numeral to decode: {self.target_numeral}\n"
            "Your goal: submit the correct decimal value this numeral represents.\n"
            "Actions (send exactly one per turn):\n"
            "  REVEAL <n>  - see how a decimal integer n (0-999 you choose) is "
            "rendered in the hidden system. Use this to probe the base.\n"
            "  MEANING <letter> - learn the digit value (0..B-1) that a single "
            "letter stands for in the hidden system.\n"
            "  GUESS_BASE <n> - declare your guess for B (one attempt, worth 0.3).\n"
            "  ANSWER <n> - declare the challenge numeral's decimal value (ends "
            "the episode, worth 0.7).\n"
            "You have 10 steps total, including submissions."
        )
        return obs, {}

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        info = {}

        text = (action or "").strip()
        m = re.match(r'^(\S+)\s*(.*)$', text)
        if not m:
            obs = "Malformed action. Use REVEAL/MEANING/GUESS_BASE/ANSWER with an argument."
            return self._finish(obs, reward, terminated, info)

        cmd = m.group(1).upper()
        arg = m.group(2).strip()

        if cmd == "REVEAL":
            if not re.match(r'^\d+$', arg):
                obs = "REVEAL needs a non-negative integer argument, e.g. REVEAL 12."
                return self._finish(obs, reward, terminated, info)
            n = int(arg)
            if n > 999:
                obs = "Pick a REVEAL value between 0 and 999."
                return self._finish(obs, reward, terminated, info)
            rendering = self._encode(n)
            obs = f"{n} renders as '{rendering}' ({len(rendering)} symbol(s))."
            return self._finish(obs, reward, terminated, info)

        elif cmd == "MEANING":
            letter = arg.strip().upper()[:1]
            if not letter or not letter.isalpha():
                obs = "MEANING needs a single letter argument, e.g. MEANING Q."
                return self._finish(obs, reward, terminated, info)
            if letter not in self.symbol_to_digit:
                obs = f"'{letter}' is not a digit symbol in this hidden system."
                return self._finish(obs, reward, terminated, info)
            obs = f"'{letter}' stands for digit value {self.symbol_to_digit[letter]}."
            return self._finish(obs, reward, terminated, info)

        elif cmd == "GUESS_BASE":
            if self.base_submitted:
                obs = "You already submitted a base guess; that milestone is closed."
                return self._finish(obs, reward, terminated, info)
            if not re.match(r'^\d+$', arg):
                obs = "GUESS_BASE needs an integer argument, e.g. GUESS_BASE 9."
                return self._finish(obs, reward, terminated, info)
            guess = int(arg)
            self.base_submitted = True
            if guess == self.base:
                reward = 0.3
                obs = "Correct base."
            else:
                obs = "Incorrect base guess."
            return self._finish(obs, reward, terminated, info)

        elif cmd == "ANSWER":
            if not re.match(r'^\d+$', arg):
                obs = "ANSWER needs an integer argument, e.g. ANSWER 42."
                terminated = True
                self.value_submitted = True
                return self._finish(obs, reward, terminated, info)
            guess = int(arg)
            self.value_submitted = True
            terminated = True
            if guess == self.target_value:
                reward = 0.7
                obs = "Correct! Episode complete."
            else:
                obs = "Incorrect final value. Episode complete."
            return self._finish(obs, reward, terminated, info)

        else:
            obs = "Unknown action. Use REVEAL, MEANING, GUESS_BASE, or ANSWER."
            return self._finish(obs, reward, terminated, info)

    def _finish(self, obs, reward, terminated, info):
        truncated = False
        if not terminated and self.steps >= self.max_steps:
            truncated = True
            obs = obs + " Step limit reached."
        return obs, reward, terminated, truncated, info
