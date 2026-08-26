import random


class CipherDiaryEnv:
    WORDS = [
        "GARDEN", "MARKET", "SILVER", "PLANET", "BRIDGE", "WINTER",
        "CASTLE", "FLIGHT", "MIRROR", "BALLOT", "PEPPER", "VOYAGE",
    ]
    GLYPH_POOL = [
        "\u2660", "\u2663", "\u2665", "\u2666", "\u2606", "\u263e",
        "\u25b2", "\u25cf", "\u25a0", "\u25c6", "\u2726", "\u2727",
        "\u25a3", "\u25d0",
    ]
    MAX_STEPS = 8
    MILESTONES = [(2, 0.15), (4, 0.20), (5, 0.25), (6, 0.40)]

    def __init__(self):
        self.rng = None
        self.secret = ""
        self.glyphs_seq = []
        self.step_count = 0
        self.best_correct = 0
        self.milestones_hit = set()
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.secret = self.rng.choice(self.WORDS)
        unique_letters = []
        for ch in self.secret:
            if ch not in unique_letters:
                unique_letters.append(ch)
        pool = self.GLYPH_POOL[:]
        self.rng.shuffle(pool)
        letter_to_glyph = {ch: pool[i] for i, ch in enumerate(unique_letters)}
        self.glyphs_seq = [letter_to_glyph[ch] for ch in self.secret]
        self.step_count = 0
        self.best_correct = 0
        self.milestones_hit = set()
        self.done = False
        n = len(self.secret)
        glyph_line = " ".join(self.glyphs_seq)
        obs = (
            "You found a locked diary. Its final line is written in a private cipher:\n"
            f"  {glyph_line}\n"
            f"This line encodes one English word of {n} letters, one letter per glyph, "
            "left to right. The SAME glyph always stands for the SAME letter, and "
            "different glyphs never stand for the same letter — repeated glyphs above "
            "are a free clue.\n"
            f"Action format: 'GUESS <letters>' with exactly {n} space-separated single "
            "letters, e.g. 'GUESS A B C D E F'.\n"
            f"After each guess you learn only HOW MANY of your {n} letters are in the "
            f"correct position (not which ones). You have {self.MAX_STEPS} steps to "
            "fully decode the word."
        )
        info = {"num_positions": n, "glyphs": list(self.glyphs_seq)}
        return obs, info

    def _parse(self, action, n):
        if not isinstance(action, str):
            return None
        tokens = action.strip().split()
        if len(tokens) != n + 1 or tokens[0].upper() != "GUESS":
            return None
        letters = []
        for t in tokens[1:]:
            if len(t) != 1 or not t.isalpha():
                return None
            letters.append(t.upper())
        return letters

    def step(self, action):
        if self.done:
            return "The diary is already fully decoded. Episode over.", 0.0, True, False, {}

        self.step_count += 1
        n = len(self.secret)
        parsed = self._parse(action, n)

        if parsed is None:
            truncated = self.step_count >= self.MAX_STEPS
            obs = (
                f"Malformed action. Use exactly: 'GUESS' followed by {n} single "
                "letters separated by spaces (e.g. 'GUESS A B C D E F'). This "
                "attempt was not scored."
            )
            if truncated:
                obs += f" You have used all {self.MAX_STEPS} steps; the diary stays locked."
                self.done = True
            return obs, 0.0, False, truncated, {"best_correct": self.best_correct}

        correct = sum(1 for g, s in zip(parsed, self.secret) if g == s)
        reward = 0.0
        if correct > self.best_correct:
            for threshold, value in self.MILESTONES:
                if correct >= threshold and threshold not in self.milestones_hit:
                    self.milestones_hit.add(threshold)
                    reward += value
            self.best_correct = correct

        terminated = correct == n
        truncated = False
        if terminated:
            self.done = True
            obs = f"Correct! {correct}/{n} letters placed — the diary is fully decoded: '{self.secret}'."
        else:
            truncated = self.step_count >= self.MAX_STEPS
            if truncated:
                self.done = True
                obs = (
                    f"{correct}/{n} letters in the correct position (best so far: "
                    f"{self.best_correct}). Out of steps — the diary remains partly locked."
                )
            else:
                remaining = self.MAX_STEPS - self.step_count
                obs = (
                    f"{correct}/{n} letters in the correct position (best so far: "
                    f"{self.best_correct}). {remaining} step(s) remain."
                )

        return obs, reward, terminated, truncated, {"correct": correct, "best_correct": self.best_correct}
