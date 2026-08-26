import random


class RotatingCipherEnv:
    """Rotating additive-shift cipher: shift_i = (s0 + i*d) mod 26.

    The solver must query a rotor oracle at chosen positions to infer the
    hidden linear shift rule, then decode the intercepted ciphertext word.
    """

    WORDLIST = [
        "PLANET", "GARDEN", "WINTER", "SPIDER", "MARKET", "SILVER", "ORANGE",
        "CASTLE", "DRAGON", "PENCIL", "VOYAGE", "TUNNEL", "HAMMER", "JACKET",
        "KITTEN", "MONKEY", "RABBIT", "SUNSET", "FOREST", "DESERT", "ISLAND",
        "BRIDGE", "CANYON", "METEOR", "ROCKET", "SPRING", "SUMMER", "AUTUMN",
        "PUZZLE", "RIDDLE", "SECRET",
    ]

    MAX_STEPS = 8
    WORD_LEN = 6

    def __init__(self):
        self.rng = None
        self.s0 = None
        self.d = None
        self.secret = None
        self.ciphertext = None
        self.steps = 0
        self.probed_positions = set()
        self.milestone_awarded = False
        self.solved = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.s0 = self.rng.randint(1, 25)
        self.d = self.rng.randint(1, 25)
        self.secret = self.rng.choice(self.WORDLIST)
        self.ciphertext = self._encrypt(self.secret)
        self.steps = 0
        self.probed_positions = set()
        self.milestone_awarded = False
        self.solved = False

        obs = (
            "ROTATING CIPHER TRACE\n"
            f"An intercepted {self.WORD_LEN}-letter word was sent through a "
            f"rotor cipher: '{self.ciphertext}'.\n"
            "Each letter position uses its own additive shift (A=0 .. Z=25), "
            "and the shifts follow one consistent hidden rule across "
            f"positions 0..{self.WORD_LEN - 1} (left to right).\n"
            "You may query the rotor: sending the reference letter 'A' "
            "through a chosen position tells you exactly what shift that "
            "position currently uses (returned as a single letter, A=0).\n"
            "ACTIONS (send exactly one per turn):\n"
            f"  PROBE <p>     - query the shift used at position p (0-{self.WORD_LEN - 1})\n"
            "  ANSWER <WORD> - submit your decoded plaintext guess\n"
            f"You have {self.MAX_STEPS} total actions. Good luck."
        )
        return obs, {"word_length": self.WORD_LEN, "max_steps": self.MAX_STEPS}

    def _shift_at(self, pos):
        return (self.s0 + pos * self.d) % 26

    def _encrypt(self, plain):
        out = []
        for i, ch in enumerate(plain):
            shift = self._shift_at(i)
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        return "".join(out)

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        text = (action or "").strip()
        parts = text.split(maxsplit=1)
        verb = parts[0].upper() if parts else ""

        if verb == "PROBE" and len(parts) == 2 and parts[1].strip().lstrip("-").isdigit():
            pos = int(parts[1].strip())
            if 0 <= pos < self.WORD_LEN:
                shift = self._shift_at(pos)
                ref_cipher = chr(shift + 65)
                self.probed_positions.add(pos)
                obs = (
                    f"Reference letter 'A' sent through position {pos} "
                    f"returned '{ref_cipher}'. (This is the raw local shift "
                    "as a letter, A=0.)"
                )
                if len(self.probed_positions) >= 2 and not self.milestone_awarded:
                    self.milestone_awarded = True
                    reward = 0.3
                    obs += " [Calibration milestone reached: +0.3]"
            else:
                obs = (
                    f"Invalid position. Choose an integer from 0 to "
                    f"{self.WORD_LEN - 1}."
                )
        elif verb == "ANSWER" and len(parts) == 2:
            guess = "".join(c for c in parts[1] if c.isalpha()).upper()
            if guess == self.secret:
                reward = 0.7
                terminated = True
                self.solved = True
                obs = f"Correct! The plaintext was '{self.secret}'."
            else:
                obs = (
                    f"Incorrect ('{guess}' does not match). The rotor state "
                    "is unchanged; you may PROBE more positions or ANSWER "
                    "again."
                )
        else:
            obs = (
                "Malformed action. Use 'PROBE <position>' or 'ANSWER <WORD>' "
                "exactly."
            )

        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            obs += " Step limit reached."

        return obs, reward, terminated, truncated, info
