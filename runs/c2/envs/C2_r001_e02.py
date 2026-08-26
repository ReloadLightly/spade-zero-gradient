import random


class AvenueGreenWaveEnv:
    CYCLE = 16
    WINDOW = 4
    N_SIGNALS = 3
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.g = []
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        max_start = self.CYCLE - self.WINDOW
        self.g = [self.rng.randint(0, max_start) for _ in range(self.N_SIGNALS)]
        self.steps = 0
        self.done = False
        obs = (
            "You are re-timing {n} traffic signals (IDs 0-{nm1}) along an avenue. Each signal "
            "has a hidden GREEN WINDOW of width {w} seconds within its {c}-second cycle "
            "(valid offsets 0-{cm1}), fixed for the whole episode. Goal: submit one offset per "
            "signal that lands inside that signal's true green window.\n"
            "Actions (exactly one per turn):\n"
            "  PROBE <signal_id> <offset>  -> reply is TOO_EARLY, TOO_LATE, or HIT for that "
            "signal's window.\n"
            "  SUBMIT <o0> <o1> ... <o{nm1}>  -> locks in one offset per signal and ends the "
            "episode; you earn {per:.4f} reward for each signal whose offset lands inside its "
            "true window (max 1.0 total).\n"
            "You have {steps} actions total, probes and the submit combined. A malformed "
            "action still costs a turn and earns no reward."
        ).format(
            n=self.N_SIGNALS, nm1=self.N_SIGNALS - 1, w=self.WINDOW, c=self.CYCLE,
            cm1=self.CYCLE - 1, per=1.0 / self.N_SIGNALS, steps=self.MAX_STEPS,
        )
        return obs, {}

    def _malformed(self, msg):
        truncated = self.steps >= self.MAX_STEPS
        self.done = truncated
        return msg, 0.0, False, truncated, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}
        self.steps += 1
        action = (action or "").strip()
        parts = action.split()

        if not parts:
            return self._malformed(
                "Malformed action. Use PROBE <signal_id> <offset> or SUBMIT <o0> ... <o{}>.".format(
                    self.N_SIGNALS - 1
                )
            )

        cmd = parts[0].upper()

        if cmd == "PROBE" and len(parts) == 3:
            sid_str, off_str = parts[1], parts[2]
            if sid_str.isdigit() and off_str.lstrip("-").isdigit():
                sid, off = int(sid_str), int(off_str)
                if 0 <= sid < self.N_SIGNALS and 0 <= off < self.CYCLE:
                    g = self.g[sid]
                    if off < g:
                        result = "TOO_EARLY"
                    elif off >= g + self.WINDOW:
                        result = "TOO_LATE"
                    else:
                        result = "HIT"
                    if self.steps >= self.MAX_STEPS:
                        self.done = True
                        return (
                            f"Signal {sid} @ offset {off}: {result}. Step budget exhausted "
                            "with no SUBMIT - episode truncated, reward 0.",
                            0.0, False, True, {},
                        )
                    return (
                        f"Signal {sid} @ offset {off}: {result}. "
                        f"Steps used: {self.steps}/{self.MAX_STEPS}.",
                        0.0, False, False, {},
                    )
            return self._malformed(
                "Malformed PROBE. Use: PROBE <signal_id 0-{}> <offset 0-{}>.".format(
                    self.N_SIGNALS - 1, self.CYCLE - 1
                )
            )

        if cmd == "SUBMIT" and len(parts) == 1 + self.N_SIGNALS:
            vals = parts[1:]
            if all(v.lstrip("-").isdigit() for v in vals):
                offsets = [int(v) for v in vals]
                if all(0 <= o < self.CYCLE for o in offsets):
                    hits = 0
                    detail = []
                    for sid, off in enumerate(offsets):
                        g = self.g[sid]
                        hit = g <= off < g + self.WINDOW
                        hits += hit
                        detail.append(
                            f"signal {sid}: {'HIT' if hit else 'MISS'} "
                            f"(true window {g}-{g + self.WINDOW - 1})"
                        )
                    reward = hits / self.N_SIGNALS
                    self.done = True
                    obs = (
                        "Final result: " + "; ".join(detail)
                        + f". {hits}/{self.N_SIGNALS} signals timed correctly."
                    )
                    return obs, reward, True, False, {}
            return self._malformed(
                "Malformed SUBMIT. Use: SUBMIT <o0> ... <o{}>, each 0-{}.".format(
                    self.N_SIGNALS - 1, self.CYCLE - 1
                )
            )

        return self._malformed(
            "Malformed action. Use PROBE <signal_id 0-{}> <offset 0-{}> or "
            "SUBMIT <o0> ... <o{}>.".format(self.N_SIGNALS - 1, self.CYCLE - 1, self.N_SIGNALS - 1)
        )
