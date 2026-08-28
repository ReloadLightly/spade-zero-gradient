import random


class HatRowStatementEnv:
    COLORS = ('R', 'G', 'B')
    COLOR_NAMES = {'R': 'RED', 'G': 'GREEN', 'B': 'BLUE'}
    COLOR_ALIASES = {
        'R': 'R', 'RED': 'R',
        'G': 'G', 'GREEN': 'G',
        'B': 'B', 'BLUE': 'B',
    }
    NUM_POSITIONS = 6
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.hats = []
        self.asked = set()
        self.locked = [False] * self.NUM_POSITIONS
        self.correct_count = 0
        self.awarded = 0.0
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.hats = [self.rng.choice(self.COLORS) for _ in range(self.NUM_POSITIONS)]
        self.asked = set()
        self.locked = [False] * self.NUM_POSITIONS
        self.correct_count = 0
        self.awarded = 0.0
        self.steps = 0
        self.done = False

        obs = (
            "GAME: Hat Row Statement Deduction\n"
            "6 players stand in a row at positions 1 (front) to 6 (back). Each "
            "wears a hidden hat colored RED, GREEN, or BLUE, chosen independently "
            "(colors may repeat). A player at position p can see every hat at "
            "positions 1..p-1 (the players ahead of them), never their own hat "
            "or the hats behind them.\n"
            "GOAL: identify all 6 hidden hat colors within 10 total actions.\n"
            "ACTIONS (exactly one per step):\n"
            "  ASK <p>    -- p in 2..6. That player truthfully states which color "
            "is the STRICT majority among the p-1 hats they see ahead of them, or "
            "'no majority' if no color has a strict majority (a tie).\n"
            "  GUESS <p> <color> -- p in 1..6, color RED/GREEN/BLUE (or R/G/B). "
            "Locks in your answer for position p. A correct guess scores 1/6 of "
            "the total reward and stays locked. A WRONG guess ends the game "
            "immediately, so guess only once you can justify it.\n"
            "You have 10 actions total; you need 6 correct guesses to win, so "
            "budget your ASKs carefully.\n"
            "Example: 'ASK 4' or 'GUESS 3 GREEN'."
        )
        return obs, {}

    def _prefix_majority(self, p):
        counts = {c: 0 for c in self.COLORS}
        for i in range(p - 1):
            counts[self.hats[i]] += 1
        best_count = max(counts.values())
        ties = [c for c in self.COLORS if counts[c] == best_count]
        if len(ties) > 1:
            return None
        return ties[0]

    def _check_truncate(self):
        if self.steps >= self.MAX_STEPS and not self.done:
            self.done = True
            return True
        return False

    def step(self, action):
        if self.done:
            return "The game has already ended.", 0.0, True, False, {}

        self.steps += 1
        parts = (action or "").strip().split()

        if not parts:
            return "Malformed action. Use 'ASK <p>' or 'GUESS <p> <color>'.", 0.0, False, self._check_truncate(), {}

        verb = parts[0].upper()

        if verb == "ASK":
            if len(parts) != 2 or not parts[1].isdigit():
                return "Malformed ASK. Use 'ASK <p>' with p an integer 2..6.", 0.0, False, self._check_truncate(), {}
            p = int(parts[1])
            if p < 2 or p > self.NUM_POSITIONS:
                return f"Position {p} cannot be asked; choose p in 2..6.", 0.0, False, self._check_truncate(), {}
            if p in self.asked:
                return f"Player {p} was already asked. Their statement is unchanged.", 0.0, False, self._check_truncate(), {}
            self.asked.add(p)
            verdict = self._prefix_majority(p)
            if verdict is None:
                obs = f"Player {p} says: 'Among the {p - 1} hats ahead of me, no color is a strict majority.'"
            else:
                obs = f"Player {p} says: 'Among the {p - 1} hats ahead of me, {self.COLOR_NAMES[verdict]} is the strict majority.'"
            return obs, 0.0, False, self._check_truncate(), {}

        if verb == "GUESS":
            if len(parts) != 3 or not parts[1].isdigit():
                return "Malformed GUESS. Use 'GUESS <p> <color>' with p an integer 1..6.", 0.0, False, self._check_truncate(), {}
            p = int(parts[1])
            color_token = parts[2].upper()
            if p < 1 or p > self.NUM_POSITIONS:
                return f"Position {p} is invalid; choose p in 1..6.", 0.0, False, self._check_truncate(), {}
            if color_token not in self.COLOR_ALIASES:
                return "Unrecognized color. Use RED, GREEN, or BLUE (or R/G/B).", 0.0, False, self._check_truncate(), {}
            if self.locked[p - 1]:
                return f"Position {p} is already locked in correctly. Choose another action.", 0.0, False, self._check_truncate(), {}

            color = self.COLOR_ALIASES[color_token]
            if color == self.hats[p - 1]:
                self.locked[p - 1] = True
                self.correct_count += 1
                if self.correct_count == self.NUM_POSITIONS:
                    reward = 1.0 - self.awarded
                    self.awarded += reward
                    self.done = True
                    obs = f"Correct! Position {p} is {self.COLOR_NAMES[color]}. All 6 hats identified — you win!"
                    return obs, reward, True, False, {}
                reward = 1.0 / self.NUM_POSITIONS
                self.awarded += reward
                obs = f"Correct! Position {p} is indeed {self.COLOR_NAMES[color]}. {self.correct_count}/6 locked in."
                return obs, reward, False, self._check_truncate(), {}
            else:
                self.done = True
                obs = (
                    f"Wrong! Position {p} is not {self.COLOR_NAMES[color]}. "
                    f"The commitment was irreversible — game over with "
                    f"{self.correct_count}/6 correctly identified before this guess."
                )
                return obs, 0.0, True, False, {}

        return "Unknown action. Use 'ASK <p>' or 'GUESS <p> <color>'.", 0.0, False, self._check_truncate(), {}
