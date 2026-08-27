import random
import string
import itertools


class PasswordRiddleEnv:
    MAX_STEPS = 10
    NUM_LETTERS = 4
    NUM_CLUES = 5

    def __init__(self):
        self.rng = None
        self.letters = []
        self.secret = {}
        self.clues = []
        self.revealed = []
        self.solved = {}
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self._generate_puzzle()
        self.revealed = [0]
        self.solved = {L: False for L in self.letters}
        self.steps = 0
        self.done = False
        obs = self._render_intro()
        info = {"letters": list(self.letters)}
        return obs, info

    def _generate_puzzle(self):
        letters_pool = list(string.ascii_uppercase)
        for _attempt in range(40):
            letters = self.rng.sample(letters_pool, self.NUM_LETTERS)
            digits = self.rng.sample(range(10), self.NUM_LETTERS)
            secret = dict(zip(letters, digits))
            candidates = self._build_candidates(letters, secret)
            self.rng.shuffle(candidates)
            chosen = candidates[: self.NUM_CLUES]
            if len(chosen) < self.NUM_CLUES:
                continue
            if self._is_unique(letters, chosen):
                self.letters = letters
                self.secret = secret
                self.clues = chosen
                return
        letters = self.rng.sample(letters_pool, self.NUM_LETTERS)
        digits = self.rng.sample(range(10), self.NUM_LETTERS)
        secret = dict(zip(letters, digits))
        clues = []
        for L in letters:
            v = secret[L]
            clues.append((f"{L} is equal to {v}.", (lambda a, L=L, v=v: a[L] == v)))
        self.letters = letters
        self.secret = secret
        self.clues = clues[: self.NUM_CLUES]

    def _build_candidates(self, letters, secret):
        cands = []
        for a, b in itertools.combinations(letters, 2):
            sa, sb = secret[a], secret[b]
            s = sa + sb
            cands.append((f"{a} plus {b} equals {s}.",
                          (lambda assign, a=a, b=b, s=s: assign[a] + assign[b] == s)))
            d = abs(sa - sb)
            cands.append((f"The difference between {a} and {b} is {d}.",
                          (lambda assign, a=a, b=b, d=d: abs(assign[a] - assign[b]) == d)))
            if sa > sb:
                cands.append((f"{a} is greater than {b}.",
                              (lambda assign, a=a, b=b: assign[a] > assign[b])))
            else:
                cands.append((f"{b} is greater than {a}.",
                              (lambda assign, a=a, b=b: assign[b] > assign[a])))
        for L in letters:
            parity = "even" if secret[L] % 2 == 0 else "odd"
            cands.append((f"{L} is an {parity} number.",
                          (lambda assign, L=L, parity=parity: (assign[L] % 2 == 0) == (parity == "even"))))
        max_letter = max(letters, key=lambda L: secret[L])
        min_letter = min(letters, key=lambda L: secret[L])
        cands.append((f"{max_letter} is the largest digit among the four.",
                      (lambda assign, max_letter=max_letter, letters=letters: assign[max_letter] == max(assign[x] for x in letters))))
        cands.append((f"{min_letter} is the smallest digit among the four.",
                      (lambda assign, min_letter=min_letter, letters=letters: assign[min_letter] == min(assign[x] for x in letters))))
        return cands

    def _is_unique(self, letters, chosen):
        count = 0
        for perm in itertools.permutations(range(10), self.NUM_LETTERS):
            assign = dict(zip(letters, perm))
            if all(pred(assign) for _, pred in chosen):
                count += 1
                if count > 1:
                    return False
        return count == 1

    def _render_intro(self):
        clue_text = self.clues[0][0]
        letters_str = ", ".join(self.letters)
        lines = [
            "You overheard a riddle about a stolen password. Four letters -- "
            f"{letters_str} -- each stand for a different digit from 0-9.",
            f"Overheard clue #1: {clue_text}",
            "",
            f"You have {self.MAX_STEPS} steps total. On each step, send one action:",
            "  ASK               -- overhear the next riddle clue (if any remain)",
            "  SOLVE X=D         -- claim that letter X stands for digit D",
            "Each correctly solved letter earns 0.25 reward (max 1.0 for all four).",
            "A letter can only earn reward once, on its first correct SOLVE.",
        ]
        return "\n".join(lines)

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        upper = action.upper()
        if upper == "ASK":
            if len(self.revealed) < len(self.clues):
                idx = len(self.revealed)
                self.revealed.append(idx)
                obs = f"Overheard clue #{idx + 1}: {self.clues[idx][0]}"
            else:
                obs = "No more clues were overheard. Nothing left to ask."
        elif upper.startswith("SOLVE"):
            parsed = self._parse_solve(action)
            if parsed is None:
                obs = "Malformed action. Use 'SOLVE X=D' with a listed letter X and a single digit D (0-9)."
            else:
                letter, digit = parsed
                if letter not in self.letters:
                    obs = f"'{letter}' is not one of the password letters ({', '.join(self.letters)})."
                elif self.solved[letter]:
                    obs = f"{letter} was already solved correctly."
                elif digit == self.secret[letter]:
                    self.solved[letter] = True
                    reward = 0.25
                    obs = f"Correct! {letter} = {digit}."
                else:
                    obs = f"Incorrect. {letter} is not {digit}. Keep deducing."
        else:
            obs = "Malformed action. Use 'ASK' or 'SOLVE X=D'."

        if all(self.solved.values()):
            terminated = True
            obs += "\nAll four letters solved! Password fully recovered."
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            remaining = [L for L in self.letters if not self.solved[L]]
            obs += f"\nStep limit reached. Unsolved letters: {', '.join(remaining)}."

        self.done = terminated or truncated
        return obs, reward, terminated, truncated, info

    def _parse_solve(self, action):
        body = action[5:].strip().replace(" ", "")
        if "=" not in body:
            return None
        letter, _, digit_str = body.partition("=")
        letter = letter.upper()
        if len(letter) != 1 or letter not in string.ascii_uppercase:
            return None
        if not digit_str.isdigit() or len(digit_str) != 1:
            return None
        return letter, int(digit_str)
