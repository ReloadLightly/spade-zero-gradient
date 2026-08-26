import random
import re


class RotatingCipherEnv:
    WORDS = ["ELEPHANT", "DOLPHINS", "PANTHERS", "TORTOISE",
             "HEDGEHOG", "LEOPARDS", "BUFFALOS", "RACCOONS"]
    MAX_STEPS = 10
    PROBE_BUDGET = 3

    def __init__(self):
        self.rng = None
        self.word = ""
        self.base = 0
        self.step_size = 1
        self.length = 0
        self.cipher = ""
        self.steps = 0
        self.probes_used = 0
        self.done = False

    def _shift_at(self, pos):
        return (self.base + pos * self.step_size) % 26

    def _encode_char(self, ch, pos):
        return chr((ord(ch) - 65 + self._shift_at(pos)) % 26 + 65)

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.word = self.rng.choice(self.WORDS)
        self.length = len(self.word)
        self.base = self.rng.randint(0, 25)
        self.step_size = self.rng.randint(1, 12)
        self.cipher = "".join(self._encode_char(c, i) for i, c in enumerate(self.word))
        self.steps = 0
        self.probes_used = 0
        self.done = False
        obs = (
            "ROTATING-KEY CIPHER. A hidden %d-letter word was enciphered letter by "
            "letter: the letter at position i (0-indexed, left to right) is shifted "
            "forward through the alphabet by some amount shift_i, and shift_i follows "
            "one consistent hidden rule as i runs from 0 to %d. You are not told the "
            "rule's form.\n"
            "Ciphertext: %s\n"
            "Actions (send exactly one per turn):\n"
            "  PROBE <letter> <position>   - ask what <letter> would encode to at that "
            "position under this message's rule (costs one of your %d probes total; "
            "does NOT reveal the hidden word directly, only that position's shift).\n"
            "  DECODE <%d-letter guess>     - submit your final plaintext guess (ends "
            "the episode; reward is the fraction of letters you get right).\n"
            "You have %d probes and %d total steps.\n"
        ) % (self.length, self.length - 1, self.cipher, self.PROBE_BUDGET,
             self.length, self.PROBE_BUDGET, self.MAX_STEPS)
        info = {"cipher": self.cipher, "length": self.length}
        return obs, info

    def step(self, action):
        if self.done:
            return "Episode already ended.", 0.0, True, False, {}

        self.steps += 1
        reward = 0.0
        terminated = False
        text = (action or "").strip()
        parts = text.split()
        verb = parts[0].upper() if parts else ""

        if verb == "PROBE" and len(parts) == 3:
            letter, pos_s = parts[1].upper(), parts[2]
            if len(letter) == 1 and letter.isalpha() and re.fullmatch(r"-?\d+", pos_s):
                pos = int(pos_s)
                if self.probes_used >= self.PROBE_BUDGET:
                    obs = "No probes remaining (%d/%d used). Try DECODE." % (
                        self.probes_used, self.PROBE_BUDGET)
                elif not (0 <= pos < self.length):
                    obs = "Invalid position %d; must be 0..%d." % (pos, self.length - 1)
                else:
                    self.probes_used += 1
                    result = self._encode_char(letter, pos)
                    obs = "PROBE: '%s' at position %d encodes to '%s'. Probes left: %d/%d." % (
                        letter, pos, result, self.PROBE_BUDGET - self.probes_used, self.PROBE_BUDGET)
            else:
                obs = "Malformed PROBE. Format: PROBE <single letter> <position integer>."
        elif verb == "DECODE" and len(parts) == 2:
            guess = parts[1].upper()
            if len(guess) == self.length and guess.isalpha():
                correct = sum(1 for a, b in zip(guess, self.word) if a == b)
                reward = correct / self.length
                terminated = True
                obs = ("DECODE received: '%s'. Correct letters: %d/%d. Actual word: %s. "
                       "Episode ended.") % (guess, correct, self.length, self.word)
            else:
                obs = "Malformed DECODE. Format: DECODE <%d-letter guess, letters only>." % self.length
        else:
            obs = ("Unrecognized action. Use PROBE <letter> <position> or "
                   "DECODE <%d-letter guess>.") % self.length

        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            obs += " Step limit reached; episode truncated."
        if terminated or truncated:
            self.done = True
        info = {"steps": self.steps, "probes_used": self.probes_used}
        return obs, reward, terminated, truncated, info
