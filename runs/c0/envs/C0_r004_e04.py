import random


class DigitSumChainEnv:
    MAX_STEPS = 10
    CHAIN_LEN = 8

    def __init__(self):
        self.rng = None
        self.terms = []
        self.corrupt_index = None
        self.corrupt_sign = None
        self.steps_taken = 0
        self.guessed_index = False
        self.guessed_sign = False
        self.done = False

    def _reverse(self, n):
        s = str(n)
        return int(s[::-1])

    def _rotate_left(self, n):
        s = str(n)
        if len(s) < 2:
            return n
        return int(s[1:] + s[0])

    def _rotate_right(self, n):
        s = str(n)
        if len(s) < 2:
            return n
        return int(s[-1] + s[:-1])

    def _swap_ends(self, n):
        s = str(n)
        if len(s) < 2:
            return n
        chars = list(s)
        chars[0], chars[-1] = chars[-1], chars[0]
        return int(''.join(chars))

    def _sort_asc(self, n):
        return int(''.join(sorted(str(n))))

    def _sort_desc(self, n):
        return int(''.join(sorted(str(n), reverse=True)))

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.steps_taken = 0
        self.done = False
        self.guessed_index = False
        self.guessed_sign = False

        seed_val = self.rng.randint(1000, 9999)
        self.terms = [seed_val]
        self.corrupt_index = self.rng.randint(1, self.CHAIN_LEN - 1)
        delta_choices = [d for d in range(-8, 9) if d != 0]
        raw_delta = self.rng.choice(delta_choices)

        transforms = [
            self._reverse, self._rotate_left, self._rotate_right,
            self._swap_ends, self._sort_asc, self._sort_desc,
        ]

        for i in range(1, self.CHAIN_LEN):
            prev = self.terms[-1]
            if i == self.corrupt_index:
                if prev + raw_delta < 1:
                    delta_actual = abs(raw_delta)
                else:
                    delta_actual = raw_delta
                new_val = prev + delta_actual
                self.corrupt_sign = 1 if delta_actual > 0 else -1
                self.terms.append(new_val)
            else:
                t = self.rng.choice(transforms)
                self.terms.append(t(prev))

        obs = (
            f"Hidden chain of {self.CHAIN_LEN} numbers, term[0..{self.CHAIN_LEN-1}].\n"
            f"term[0] = {seed_val}.\n"
            f"Each term[i] (i=1..{self.CHAIN_LEN-1}) was produced from term[i-1] by one hidden "
            f"operation. Most operations just rearrange the digits of the previous number; "
            f"exactly ONE step among 1..{self.CHAIN_LEN-1} instead perturbs the number by adding "
            f"or subtracting a small nonzero amount, breaking whatever regularity the "
            f"rearrangements preserve. Find that corrupted step's index and the sign of its "
            f"perturbation.\n"
            f"Actions (one per line):\n"
            f"  PROBE <i>        reveal term[i] for i in 1..{self.CHAIN_LEN-1}\n"
            f"  GUESS_INDEX <i>  commit your answer for the corrupted step index\n"
            f"  GUESS_SIGN <+1 or -1>  commit whether the perturbation added (+1) or subtracted (-1)\n"
            f"You have {self.MAX_STEPS} steps total. The episode ends once you submit both "
            f"guesses, or when steps run out."
        )
        return obs, {}

    def step(self, action):
        self.steps_taken += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        parts = action.strip().split()
        if not parts:
            obs = "Malformed action. Use: PROBE <i>, GUESS_INDEX <i>, or GUESS_SIGN <+1/-1>."
        else:
            cmd = parts[0].upper()
            if cmd == "PROBE" and len(parts) == 2 and parts[1].lstrip('-').isdigit():
                i = int(parts[1])
                if 1 <= i <= self.CHAIN_LEN - 1:
                    obs = f"term[{i}] = {self.terms[i]}"
                else:
                    obs = f"Invalid index {i}. Valid PROBE indices are 1..{self.CHAIN_LEN-1}."
            elif cmd == "GUESS_INDEX" and len(parts) == 2 and parts[1].lstrip('-').isdigit():
                if self.guessed_index:
                    obs = "You already submitted GUESS_INDEX."
                else:
                    i = int(parts[1])
                    self.guessed_index = True
                    if i == self.corrupt_index:
                        reward += 0.6
                        obs = f"Correct: the corrupted step is index {i}."
                    else:
                        obs = f"Incorrect. Index {i} is not the corrupted step."
            elif cmd == "GUESS_SIGN" and len(parts) == 2 and parts[1] in ("+1", "-1"):
                if self.guessed_sign:
                    obs = "You already submitted GUESS_SIGN."
                else:
                    s = 1 if parts[1] == "+1" else -1
                    self.guessed_sign = True
                    if s == self.corrupt_sign:
                        reward += 0.4
                        obs = f"Correct: the perturbation sign was {parts[1]}."
                    else:
                        obs = f"Incorrect. The sign was not {parts[1]}."
            else:
                obs = (
                    "Malformed action. Use: PROBE <i> (i=1.."
                    f"{self.CHAIN_LEN-1}), GUESS_INDEX <i>, or GUESS_SIGN <+1/-1>."
                )

        if self.guessed_index and self.guessed_sign:
            terminated = True
            self.done = True
        if not terminated and self.steps_taken >= self.MAX_STEPS:
            truncated = True
            self.done = True

        obs += f"\nSteps used: {self.steps_taken}/{self.MAX_STEPS}."
        return obs, reward, terminated, truncated, info
