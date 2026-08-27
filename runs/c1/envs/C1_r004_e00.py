import random
import itertools


class PasswordRiddleEnv:
    """Recover a hidden 4-digit password by querying overheard riddle clues."""

    NUM_POSITIONS = 4
    DIGIT_POOL = list(range(1, 8))  # 1..7
    MAX_STEPS = 10
    NUM_CLUES = 6

    def __init__(self):
        self.rng = None
        self.digits = []
        self.clue_text = {}
        self.heard = set()
        self.step_count = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self._generate_episode()
        self.heard = set()
        self.step_count = 0
        self.done = False
        topics = (
            "1: the sum of all four digits\n"
            "2: how many of the four digits are even\n"
            "3: which position holds the largest digit\n"
            "4: which position holds the smallest digit\n"
            "5: how the two non-extreme positions compare to each other\n"
            "6: the product of the largest and smallest digit"
        )
        obs = (
            "A vault password is a secret arrangement of 4 distinct digits "
            "(each from 1-7) across positions 1-4, left to right. You have "
            "overheard witnesses who each know one fact about it. You have "
            f"{self.MAX_STEPS} total steps.\n"
            "Actions:\n"
            "  LISTEN <n>  -- ask witness n (1-6) to repeat their fact "
            "(each witness answers only once)\n"
            "  GUESS d1 d2 d3 d4  -- submit your final password guess "
            "(four distinct digits 1-7); this ends the episode\n"
            "Available witness topics:\n" + topics
        )
        return obs, {}

    def _compute_clues(self, digits):
        total = sum(digits)
        evens = sum(1 for d in digits if d % 2 == 0)
        max_val = max(digits)
        min_val = min(digits)
        max_pos = digits.index(max_val) + 1
        min_pos = digits.index(min_val) + 1
        remaining = sorted(
            p for p in range(1, self.NUM_POSITIONS + 1)
            if p not in (max_pos, min_pos)
        )
        pos_a, pos_b = remaining
        cmp_gt = digits[pos_a - 1] > digits[pos_b - 1]
        product = max_val * min_val
        return (total, evens, max_pos, min_pos, pos_a, pos_b, cmp_gt, product)

    def _generate_episode(self):
        candidate = None
        target = None
        for _ in range(500):
            candidate = self.rng.sample(self.DIGIT_POOL, self.NUM_POSITIONS)
            target = self._compute_clues(candidate)
            matches = 0
            for perm in itertools.permutations(self.DIGIT_POOL, self.NUM_POSITIONS):
                if self._compute_clues(list(perm)) == target:
                    matches += 1
                    if matches > 1:
                        break
            if matches == 1:
                break
        self.digits = candidate
        self._build_clue_text(target)

    def _build_clue_text(self, computed):
        total, evens, max_pos, min_pos, pos_a, pos_b, cmp_gt, product = computed
        cmp_word = "greater than" if cmp_gt else "less than"
        self.clue_text = {
            1: f"The four digits summed to {total}.",
            2: f"Exactly {evens} of the four digits were even.",
            3: f"The largest digit sat in position {max_pos}.",
            4: f"The smallest digit sat in position {min_pos}.",
            5: (
                f"Between position {pos_a} and position {pos_b}, the digit "
                f"in position {pos_a} was {cmp_word} the digit in position {pos_b}."
            ),
            6: f"The product of the largest and smallest digit was {product}.",
        }

    def step(self, action):
        if self.done:
            return "The episode has already ended.", 0.0, True, False, {}

        action = (action or "").strip()
        tokens = action.split()
        reward = 0.0
        terminated = False

        if tokens and tokens[0].upper() == "LISTEN" and len(tokens) == 2:
            obs, reward, terminated = self._handle_listen(tokens[1])
        elif tokens and tokens[0].upper() == "GUESS" and len(tokens) == 5:
            obs, reward, terminated = self._handle_guess(tokens[1:])
        else:
            obs = (
                "Unrecognized action. Use 'LISTEN <n>' with n from 1-6, or "
                "'GUESS d1 d2 d3 d4' with four distinct digits from 1-7."
            )

        self.step_count += 1
        truncated = False
        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated = True
        self.done = terminated or truncated
        return obs, reward, terminated, truncated, {}

    def _handle_listen(self, arg):
        if not arg.isdigit() or not (1 <= int(arg) <= self.NUM_CLUES):
            valid = [n for n in range(1, self.NUM_CLUES + 1) if n not in self.heard]
            return (
                f"Invalid witness number. Unheard witnesses: {valid}.",
                0.0,
                False,
            )
        n = int(arg)
        if n in self.heard:
            return (
                f"Witness {n} already spoke to you and has nothing new to say.",
                0.0,
                False,
            )
        self.heard.add(n)
        return (f"Witness {n} recalls: \"{self.clue_text[n]}\"", 0.0, False)

    def _handle_guess(self, digit_tokens):
        if not all(t.isdigit() for t in digit_tokens):
            return (
                "Guess must be four space-separated digits, e.g. 'GUESS 3 1 5 2'.",
                0.0,
                False,
            )
        guess = [int(t) for t in digit_tokens]
        if len(set(guess)) != 4 or any(d not in self.DIGIT_POOL for d in guess):
            return (
                "Guess must be four distinct digits, each from 1-7.",
                0.0,
                False,
            )
        correct_positions = sum(1 for i in range(4) if guess[i] == self.digits[i])
        reward = 0.2 * correct_positions
        if correct_positions == self.NUM_POSITIONS:
            reward += 0.2
            obs = f"Correct! The password was {self.digits}. The vault opens."
        else:
            obs = (
                f"Wrong. {correct_positions}/4 digits were in the correct "
                f"position. The vault stays sealed. The true password was "
                f"{self.digits}."
            )
        return obs, reward, True
