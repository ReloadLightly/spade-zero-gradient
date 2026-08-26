import random
import re
import string


class MosaicBorderEnv:
    def __init__(self):
        self.rng = None
        self.colors = string.ascii_uppercase[:6]
        self.period = 0
        self.shift = 0
        self.motif = []
        self.visible_len = 10
        self.hidden_span = 8
        self.total_len = self.visible_len + self.hidden_span
        self.checkpoints = []
        self.resolved = {}
        self.steps = 0
        self.max_steps = 10
        self.done = False

    def _tile(self, i):
        m = self.motif[i % self.period]
        base = self.colors.index(m)
        idx = (base + self.shift * (i // self.period)) % len(self.colors)
        return self.colors[idx]

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.period = self.rng.randint(3, 5)
        self.shift = self.rng.randint(1, 5)
        self.motif = [self.rng.choice(self.colors) for _ in range(self.period)]
        self.total_len = self.visible_len + self.hidden_span
        offsets = sorted(self.rng.sample(range(1, self.hidden_span + 1), 4))
        self.checkpoints = [self.visible_len + o - 1 for o in offsets]
        self.resolved = {}
        self.steps = 0
        self.done = False

        visible = [self._tile(i) for i in range(self.visible_len)]
        cp_positions = ', '.join(str(c + 1) for c in self.checkpoints)
        peekable_positions = ', '.join(
            str(i + 1) for i in range(self.visible_len, self.total_len)
            if i not in self.checkpoints
        )
        obs = (
            "MOSAIC BORDER PATTERN\n"
            f"Colors cycle in this fixed order (wraps around): {' -> '.join(self.colors)} -> ...\n"
            f"Tiles at positions 1-{self.visible_len} of the border are:\n"
            f"{' '.join(visible)}\n"
            "The same underlying rule continues beyond what you can see.\n"
            f"You must predict the color at TARGET positions: {cp_positions}.\n"
            f"Positions you may inspect first (not targets): {peekable_positions}.\n"
            "ACTIONS (one per turn):\n"
            "  peek <position>            - reveal a non-target hidden tile (costs a step, no reward)\n"
            "  predict <position> <letter> - lock in a guess for one target position (one attempt each)\n"
            f"You have {self.max_steps} steps total. Each correct target prediction is worth partial reward; "
            "all four correct gives full reward.\n"
        )
        return obs, {"total_length": self.total_len}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {"steps": self.steps}

        self.steps += 1
        reward = 0.0
        match = re.match(r'^\s*(peek|predict)\s+(\d+)(?:\s+([A-Za-z]))?\s*$', action or '', re.IGNORECASE)

        if not match:
            obs = ("Malformed action. Use 'peek <position>' or 'predict <position> <letter>' "
                   "with a numeric position and, for predict, a single color letter.")
        else:
            cmd = match.group(1).lower()
            pos1 = int(match.group(2))
            letter = match.group(3)
            idx0 = pos1 - 1

            if idx0 < 0 or idx0 >= self.total_len:
                obs = f"Position {pos1} is out of range (1-{self.total_len})."
            elif cmd == 'peek':
                if idx0 in self.checkpoints:
                    obs = f"Position {pos1} is a TARGET position; use predict there, not peek."
                else:
                    obs = f"Position {pos1} is '{self._tile(idx0)}'."
            else:
                if idx0 not in self.checkpoints:
                    obs = f"Position {pos1} is not a target position; predict is only valid there."
                elif idx0 in self.resolved:
                    obs = f"Position {pos1} was already resolved; no more attempts allowed there."
                elif letter is None:
                    obs = "Predict requires a single color letter, e.g. 'predict 12 C'."
                else:
                    letter = letter.upper()
                    if letter not in self.colors:
                        obs = f"'{letter}' is not a valid color; use one of {', '.join(self.colors)}."
                    else:
                        correct = (letter == self._tile(idx0))
                        self.resolved[idx0] = correct
                        if correct:
                            reward = 0.25
                            obs = f"Position {pos1}: correct."
                        else:
                            obs = f"Position {pos1}: incorrect."

        remaining = self.max_steps - self.steps
        terminated = len(self.resolved) == 4
        truncated = (not terminated) and self.steps >= self.max_steps
        self.done = terminated or truncated
        correct_count = sum(1 for v in self.resolved.values() if v)
        obs = obs + f" Resolved {len(self.resolved)}/4, {correct_count} correct. Steps remaining: {max(remaining, 0)}."

        info = {"steps": self.steps, "resolved": len(self.resolved), "correct": correct_count}
        return obs, reward, terminated, truncated, info
