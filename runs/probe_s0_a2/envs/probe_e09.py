import random
import re
import itertools
from collections import Counter


class LockNotesEnv:
    def __init__(self):
        self.max_steps = 10
        self.rng = None
        self.secret = None
        self.clue_texts = []
        self.clue_preds = []
        self.step_count = 0
        self.best_correct = 0
        self.finished = False
        self._action_re = re.compile(r'^\s*(TEST|SUBMIT)\s+(\d{4})\s*$', re.IGNORECASE)

    def _gen_sum(self, rng, secret):
        s = sum(secret)
        return (f"the four digits sum to {s}",
                lambda combo, s=s: sum(combo) == s)

    def _gen_even_count(self, rng, secret):
        k = sum(1 for d in secret if d % 2 == 0)
        return (f"exactly {k} of the four digits are even numbers",
                lambda combo, k=k: sum(1 for d in combo if d % 2 == 0) == k)

    def _gen_pos_parity(self, rng, secret):
        pos = rng.randrange(4)
        parity = "even" if secret[pos] % 2 == 0 else "odd"
        return (f"the digit in position {pos + 1} is {parity}",
                lambda combo, pos=pos, parity=parity: (combo[pos] % 2 == 0) == (parity == "even"))

    def _gen_compare(self, rng, secret):
        i, j = rng.sample(range(4), 2)
        if secret[i] > secret[j]:
            rel = "greater than"
        elif secret[i] < secret[j]:
            rel = "less than"
        else:
            rel = "equal to"
        def pred(combo, i=i, j=j, rel=rel):
            if rel == "greater than":
                return combo[i] > combo[j]
            if rel == "less than":
                return combo[i] < combo[j]
            return combo[i] == combo[j]
        return (f"the digit in position {i + 1} is {rel} the digit in position {j + 1}", pred)

    def _gen_max(self, rng, secret):
        m = max(secret)
        return (f"the largest digit in the combination is {m}",
                lambda combo, m=m: max(combo) == m)

    def _gen_min(self, rng, secret):
        m = min(secret)
        return (f"the smallest digit in the combination is {m}",
                lambda combo, m=m: min(combo) == m)

    def _gen_distinct(self, rng, secret):
        d = len(set(secret))
        return (f"the combination contains exactly {d} distinct digit value(s)",
                lambda combo, d=d: len(set(combo)) == d)

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        rng = self.rng
        self.secret = [rng.randint(0, 9) for _ in range(4)]
        self.step_count = 0
        self.best_correct = 0
        self.finished = False

        kind_funcs = {
            'sum': self._gen_sum,
            'even_count': self._gen_even_count,
            'pos_parity': self._gen_pos_parity,
            'compare': self._gen_compare,
            'max': self._gen_max,
            'min': self._gen_min,
            'distinct': self._gen_distinct,
        }
        kinds = list(kind_funcs.keys())
        all_combos = list(itertools.product(range(10), repeat=4))

        best_attempt = None
        target_mid = 20
        for _ in range(40):
            chosen = rng.sample(kinds, 3)
            texts, preds = [], []
            for k in chosen:
                text, pred = kind_funcs[k](rng, self.secret)
                texts.append(text)
                preds.append(pred)
            count = sum(1 for c in all_combos if all(p(c) for p in preds))
            score = abs(count - target_mid)
            if best_attempt is None or score < best_attempt[3]:
                best_attempt = (texts, preds, count, score)
            if 4 <= count <= 60:
                break
        texts, preds, count, _ = best_attempt
        self.clue_texts = texts
        self.clue_preds = preds

        lines = [f"Clue note {i + 1}: {t}." for i, t in enumerate(self.clue_texts)]
        obs = (
            "You must reconstruct a 4-digit lock combination (each digit 0-9, "
            "digits may repeat) from torn clue notes.\n"
            + "\n".join(lines) + "\n\n"
            "Action format (send exactly one per turn):\n"
            "  TEST DDDD   - press a candidate combination against the lock's "
            "diagnostic pins. It reports how many digits are in the exactly "
            "correct position, and how many correct digits are present but in "
            "the wrong position. It does NOT end the episode.\n"
            "  SUBMIT DDDD - your real attempt to open the lock with this "
            "combination. This ENDS the episode whether or not it is correct.\n"
            f"You have {self.max_steps} actions total (TEST and SUBMIT both count "
            "against this limit). Example: TEST 4071"
        )
        return obs, {"max_steps": self.max_steps}

    def _score(self, guess):
        correct_pos = sum(g == s for g, s in zip(guess, self.secret))
        gc = Counter(guess)
        sc = Counter(self.secret)
        total_match = sum(min(gc[d], sc[d]) for d in gc)
        wrong_pos = total_match - correct_pos
        return correct_pos, wrong_pos

    def step(self, action):
        if self.finished:
            return ("Episode already ended.", 0.0, True, False, {})

        self.step_count += 1
        match = self._action_re.match(action or "")

        if not match:
            obs = (
                "Malformed action. Use exactly 'TEST DDDD' or 'SUBMIT DDDD' "
                "where DDDD is four digits, e.g. TEST 4071."
            )
            truncated = self.step_count >= self.max_steps
            if truncated:
                self.finished = True
            return (obs, 0.0, False, truncated, {"step": self.step_count})

        verb = match.group(1).upper()
        guess = [int(ch) for ch in match.group(2)]
        correct_pos, wrong_pos = self._score(guess)

        increase = max(0, correct_pos - self.best_correct)
        reward = (increase / 4.0) * 0.6
        self.best_correct = max(self.best_correct, correct_pos)

        if verb == "TEST":
            obs = (
                f"Diagnostic result for {''.join(map(str, guess))}: "
                f"{correct_pos} digit(s) in the exactly correct position; "
                f"{wrong_pos} more correct digit(s) present but in the wrong position."
            )
            truncated = self.step_count >= self.max_steps
            if truncated:
                self.finished = True
            info = {"step": self.step_count, "correct_positions": correct_pos,
                     "wrong_position_matches": wrong_pos}
            return (obs, reward, False, truncated, info)

        # SUBMIT
        self.finished = True
        if correct_pos == 4:
            reward += 0.4
            obs = f"The lock opens with {''.join(map(str, guess))}. Success."
        else:
            obs = (
                f"The lock does not open. {''.join(map(str, guess))} had "
                f"{correct_pos} digit(s) in the correct position and "
                f"{wrong_pos} correct digit(s) in the wrong position. "
                "That was your real attempt; the episode has ended."
            )
        info = {"step": self.step_count, "correct_positions": correct_pos,
                "wrong_position_matches": wrong_pos, "secret": self.secret}
        return (obs, reward, True, False, info)
