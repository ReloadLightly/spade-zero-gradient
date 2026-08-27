import random


class DrumRotationAccentEnv:
    L = 6
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.p = self.rng.randrange(self.L)
        base_positions = {self.p, (self.p + 3) % self.L}
        self.p_acc = self.rng.choice(sorted(base_positions))
        self.s = self.rng.choice([1, 2, 4, 5])
        self.steps = 0
        self.done = False
        self.awarded_probe_milestone = False

        self.bar_strs = {n: self._bar_string(n) for n in (1, 2, 3)}

        obs = (
            "DRUM ROTATION DETECTIVE\n"
            f"A {self.L}-step drum bar rotates by a fixed hidden shift each bar. "
            "Every bar has exactly 2 hits, always 3 steps apart (positions p and p+3 mod 6).\n"
            "Bar 1: " + self.bar_strs[1] + "\n"
            "Bar 2: " + self.bar_strs[2] + "\n"
            "Bar 3: " + self.bar_strs[3] + "\n"
            "(X = hit, . = rest; positions read left-to-right as 0..5)\n"
            "One of the two hits in each bar is secretly ACCENTED (played strong); the other is weak. "
            "This is never shown directly -- you must probe for it.\n"
            "GOAL: determine Bar 6's two hit positions and which one is accented.\n"
            "ACTIONS:\n"
            "  ACCENT <bar> <pos>  - ask whether position pos (0-5) in bar (1-5) is STRONG, WEAK, or NO HIT\n"
            "  COMMIT <pos1> <pos2> <accent_pos>  - submit Bar 6's two hit positions and the accented one; ends the episode\n"
            f"You have {self.MAX_STEPS} actions total (probes + the commit). Malformed actions are corrected and cost no action."
        )
        return obs, {}

    def _positions(self, n):
        base_q = (self.p + (n - 1) * self.s) % self.L
        return {base_q, (base_q + 3) % self.L}

    def _accent(self, n):
        return (self.p_acc + (n - 1) * self.s) % self.L

    def _bar_string(self, n):
        pos = self._positions(n)
        return ''.join('X' if i in pos else '.' for i in range(self.L))

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        text = (action or "").strip().upper()
        parts = text.split()

        if not parts:
            return ("Empty action. Use ACCENT <bar> <pos> or COMMIT <pos1> <pos2> <accent_pos>.",
                    0.0, False, False, {})

        cmd = parts[0]

        if cmd == "ACCENT":
            if len(parts) != 3 or not parts[1].lstrip('-').isdigit() or not parts[2].lstrip('-').isdigit():
                return "Malformed. Use: ACCENT <bar 1-5> <pos 0-5>.", 0.0, False, False, {}
            bar = int(parts[1])
            pos = int(parts[2])
            if not (1 <= bar <= 5) or not (0 <= pos < self.L):
                return "Out of range. bar must be 1-5, pos must be 0-5.", 0.0, False, False, {}

            self.steps += 1
            hits = self._positions(bar)
            reward = 0.0
            if pos not in hits:
                obs = f"Bar {bar} pos {pos}: NO HIT. ({self.steps}/{self.MAX_STEPS} actions used)"
            else:
                accented = self._accent(bar)
                label = "STRONG" if pos == accented else "WEAK"
                obs = f"Bar {bar} pos {pos}: {label}. ({self.steps}/{self.MAX_STEPS} actions used)"
                if bar in (2, 4) and not self.awarded_probe_milestone:
                    reward += 0.2
                    self.awarded_probe_milestone = True

            if self.steps >= self.MAX_STEPS:
                self.done = True
                return obs + " Step limit reached without a COMMIT.", reward, False, True, {}
            return obs, reward, False, False, {}

        if cmd == "COMMIT":
            if len(parts) != 4 or not all(t.lstrip('-').isdigit() for t in parts[1:]):
                return "Malformed. Use: COMMIT <pos1> <pos2> <accent_pos>.", 0.0, False, False, {}
            pos1, pos2, acc = (int(t) for t in parts[1:])
            self.steps += 1
            self.done = True

            true_positions = self._positions(6)
            true_accent = self._accent(6)

            reward = 0.0
            if {pos1, pos2} == true_positions:
                reward += 0.4
            if acc == true_accent and acc in {pos1, pos2}:
                reward += 0.4

            total = reward + (0.2 if self.awarded_probe_milestone else 0.0)
            obs = (
                f"COMMIT received: positions {{{pos1},{pos2}}}, accent {acc}. "
                f"Bar 6 actually: {self._bar_string(6)} with accent at {true_accent}. "
                f"Final reward this episode: {total:.2f}"
            )
            return obs, reward, True, False, {}

        return "Unknown command. Use ACCENT or COMMIT.", 0.0, False, False, {}
