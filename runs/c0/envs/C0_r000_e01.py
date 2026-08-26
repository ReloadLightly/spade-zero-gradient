import random
import re


class RotatingCipherEnv:
    WORDS = [
        "GARDEN", "PLANET", "SILVER", "WINTER", "CASTLE", "DRAGON", "BRIDGE",
        "FOREST", "MIRROR", "MARBLE", "PENCIL", "ROCKET", "SPIDER", "TUNNEL",
        "WALNUT", "YELLOW", "ORANGE", "PURPLE", "VIOLET", "ISLAND",
    ]
    D_CHOICES = [1, 3, 5, 7, 9, 11]
    WORD_LEN = 6
    MAX_STEPS = 10
    MAX_PROBES = 3

    def __init__(self):
        self.rng = None
        self.word = None
        self.cipher = None
        self.s0 = None
        self.d = None
        self.step_count = 0
        self.probes_used = 0
        self.revealed = {}
        self.key_reward_given = False
        self.terminated = False
        self.truncated = False

    def _encrypt(self, word, s0, d):
        out = []
        for i, ch in enumerate(word):
            shift = (s0 + i * d) % 26
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        return "".join(out)

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.word = self.rng.choice(self.WORDS)
        self.s0 = self.rng.randint(0, 25)
        self.d = self.rng.choice(self.D_CHOICES)
        self.cipher = self._encrypt(self.word, self.s0, self.d)
        self.step_count = 0
        self.probes_used = 0
        self.revealed = {}
        self.key_reward_given = False
        self.terminated = False
        self.truncated = False
        obs = (
            "You are cracking a rotating substitution cipher.\n"
            "Each plaintext letter at position i (0-indexed) was shifted by "
            "(s0 + i*d) mod 26 to make the ciphertext letter. s0 and d are fixed, "
            f"unknown integers: s0 in 0..25, d in {{1,3,5,7,9,11}}.\n"
            f"CIPHERTEXT ({self.WORD_LEN} letters): {self.cipher}\n"
            "Actions (exactly one per turn):\n"
            f"  PROBE <i>    - reveal the true plaintext letter at position i "
            f"(0..{self.WORD_LEN - 1}); limited to {self.MAX_PROBES} uses total.\n"
            "  KEY <s0> <d> - submit your guess for the two key integers.\n"
            f"  SOLVE <WORD> - submit your final {self.WORD_LEN}-letter plaintext "
            "guess; this ends the episode.\n"
            f"You have {self.MAX_STEPS} steps total. Reward: 0.4 for the correct "
            "KEY, 0.6 for the correct SOLVE."
        )
        return obs, {}

    def step(self, action):
        if self.terminated or self.truncated:
            return "Episode already ended.", 0.0, self.terminated, self.truncated, {}

        self.step_count += 1
        act = (action or "").strip()
        reward = 0.0
        obs = ""

        m = re.match(r'^PROBE\s+(-?\d+)$', act, re.IGNORECASE)
        if m:
            i = int(m.group(1))
            if self.probes_used >= self.MAX_PROBES:
                obs = f"No probes remaining (used {self.probes_used}/{self.MAX_PROBES})."
            elif i < 0 or i >= self.WORD_LEN:
                obs = f"Invalid position {i}; must be 0..{self.WORD_LEN - 1}. Step still counted."
            else:
                self.probes_used += 1
                self.revealed[i] = self.word[i]
                obs = (
                    f"Position {i}: plaintext='{self.word[i]}', "
                    f"ciphertext='{self.cipher[i]}' "
                    f"(probes used: {self.probes_used}/{self.MAX_PROBES})."
                )
        else:
            m = re.match(r'^KEY\s+(-?\d+)\s+(-?\d+)$', act, re.IGNORECASE)
            if m:
                gs0 = int(m.group(1)) % 26
                gd = int(m.group(2))
                matches = 0
                for i, p in self.revealed.items():
                    true_shift = (ord(self.cipher[i]) - ord(p)) % 26
                    guess_shift = (gs0 + i * gd) % 26
                    if guess_shift == true_shift:
                        matches += 1
                if gs0 == self.s0 and gd == self.d:
                    if not self.key_reward_given:
                        reward = 0.4
                        self.key_reward_given = True
                        obs = f"Correct key! s0={gs0}, d={gd}."
                    else:
                        obs = f"Key already confirmed correct (s0={gs0}, d={gd})."
                elif self.revealed:
                    obs = f"Incorrect key. Consistent with {matches}/{len(self.revealed)} revealed position(s)."
                else:
                    obs = "Incorrect key. Probe some positions to get feedback."
            else:
                m = re.match(r'^SOLVE\s+([A-Za-z]+)$', act, re.IGNORECASE)
                if m:
                    guess = m.group(1).upper()
                    if guess == self.word:
                        reward = 0.6
                        obs = f"Correct! The plaintext was {self.word}."
                    else:
                        correct_letters = sum(1 for a, b in zip(guess, self.word) if a == b)
                        obs = (
                            f"Incorrect. '{guess}' matched {correct_letters}/{self.WORD_LEN} "
                            f"letters in the true word (true length {self.WORD_LEN})."
                        )
                    self.terminated = True
                else:
                    obs = "Malformed action. Use 'PROBE <i>', 'KEY <s0> <d>', or 'SOLVE <WORD>'."

        if not self.terminated and self.step_count >= self.MAX_STEPS:
            self.truncated = True

        info = {"step": self.step_count, "probes_used": self.probes_used}
        return obs, reward, self.terminated, self.truncated, info
