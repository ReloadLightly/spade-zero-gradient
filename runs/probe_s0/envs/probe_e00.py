import random
import re


class TreasureGridEnv:
    GRID_SIZE = 8
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.treasure_row = None
        self.treasure_col = None
        self.row_candidates = None
        self.col_candidates = None
        self.row_solved_rewarded = False
        self.col_solved_rewarded = False
        self.steps_taken = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.treasure_row = self.rng.randint(1, self.GRID_SIZE)
        self.treasure_col = self.rng.randint(1, self.GRID_SIZE)
        self.row_candidates = set(range(1, self.GRID_SIZE + 1))
        self.col_candidates = set(range(1, self.GRID_SIZE + 1))
        self.row_solved_rewarded = False
        self.col_solved_rewarded = False
        self.steps_taken = 0
        self.done = False

        obs = (
            "A pirate's map divides the island into an 8x8 grid: rows 1 (northmost) "
            "to 8 (southmost), columns 1 (westmost) to 8 (eastmost). The treasure sits "
            "at one hidden (row, col) cell. You have 10 total actions to find it.\n"
            "Actions (send exactly one per turn):\n"
            "  PROBE ROW <n>     -- learn whether the treasure is north/south of row n, or an exact match\n"
            "  PROBE COL <n>     -- learn whether the treasure is west/east of column n, or an exact match\n"
            "  GUESS <row> <col> -- claim the exact cell; a correct claim ends the game\n"
            "n must be an integer from 1 to 8. You win by sending a correct GUESS."
        )
        info = {"row_candidates": len(self.row_candidates), "col_candidates": len(self.col_candidates)}
        return obs, info

    def _corrective(self, msg):
        self.steps_taken += 1
        truncated = self.steps_taken >= self.MAX_STEPS and not self.done
        return msg, 0.0, False, truncated, {"steps_remaining": self.MAX_STEPS - self.steps_taken}

    def step(self, action):
        if self.done:
            return "The episode has already ended.", 0.0, True, False, {}

        text = (action or "").strip().upper()
        m_probe = re.fullmatch(r"PROBE\s+(ROW|COL)\s+(-?\d+)", text)
        m_guess = re.fullmatch(r"GUESS\s+(-?\d+)\s+(-?\d+)", text)

        if m_probe:
            axis, n_str = m_probe.groups()
            n = int(n_str)
            if not (1 <= n <= self.GRID_SIZE):
                return self._corrective(
                    f"Malformed probe: {n} is outside 1-{self.GRID_SIZE}. "
                    "Use PROBE ROW <n> or PROBE COL <n> with n in 1-8."
                )
            self.steps_taken += 1
            reward = 0.0

            if axis == "ROW":
                target = self.treasure_row
                if n == target:
                    self.row_candidates &= {n}
                    obs = f"Exact match! The treasure's row is {n}."
                elif n < target:
                    self.row_candidates = {r for r in self.row_candidates if r > n}
                    obs = f"The treasure is south of row {n}."
                else:
                    self.row_candidates = {r for r in self.row_candidates if r < n}
                    obs = f"The treasure is north of row {n}."
                if len(self.row_candidates) == 1 and not self.row_solved_rewarded:
                    reward += 0.25
                    self.row_solved_rewarded = True
                    obs += " (Row narrowed to a single possibility.)"
                obs += (
                    f" [{len(self.row_candidates)} row candidate(s), "
                    f"{len(self.col_candidates)} col candidate(s) remain]"
                )
            else:
                target = self.treasure_col
                if n == target:
                    self.col_candidates &= {n}
                    obs = f"Exact match! The treasure's column is {n}."
                elif n < target:
                    self.col_candidates = {c for c in self.col_candidates if c > n}
                    obs = f"The treasure is east of column {n}."
                else:
                    self.col_candidates = {c for c in self.col_candidates if c < n}
                    obs = f"The treasure is west of column {n}."
                if len(self.col_candidates) == 1 and not self.col_solved_rewarded:
                    reward += 0.25
                    self.col_solved_rewarded = True
                    obs += " (Column narrowed to a single possibility.)"
                obs += (
                    f" [{len(self.row_candidates)} row candidate(s), "
                    f"{len(self.col_candidates)} col candidate(s) remain]"
                )

            truncated = self.steps_taken >= self.MAX_STEPS
            info = {"row_candidates": len(self.row_candidates), "col_candidates": len(self.col_candidates)}
            return obs, reward, False, truncated, info

        if m_guess:
            r_str, c_str = m_guess.groups()
            r, c = int(r_str), int(c_str)
            if not (1 <= r <= self.GRID_SIZE and 1 <= c <= self.GRID_SIZE):
                return self._corrective(
                    f"Malformed guess: coordinates must be integers in 1-{self.GRID_SIZE}."
                )
            self.steps_taken += 1
            if r == self.treasure_row and c == self.treasure_col:
                self.done = True
                return (
                    f"Correct! The treasure was at ({r}, {c}). You found it.",
                    0.5,
                    True,
                    False,
                    {"row_candidates": 1, "col_candidates": 1},
                )
            hint_parts = [
                "row correct" if r == self.treasure_row else "row wrong",
                "col correct" if c == self.treasure_col else "col wrong",
            ]
            truncated = self.steps_taken >= self.MAX_STEPS
            obs = f"Wrong guess ({r}, {c}): {', '.join(hint_parts)}. Keep probing or guess again."
            info = {"row_candidates": len(self.row_candidates), "col_candidates": len(self.col_candidates)}
            return obs, 0.0, False, truncated, info

        return self._corrective(
            "Unrecognized action. Use 'PROBE ROW <n>', 'PROBE COL <n>', or 'GUESS <row> <col>' with n in 1-8."
        )
