import random


class GreenWaveTimingEnv:
    MAX_STEPS = 10
    TOLERANCE = 4

    def __init__(self):
        self.rng = None
        self.cycle = 0
        self.green_len = 0
        self.unknown_ids = []
        self.targets = {}
        self.locked = {}
        self.steps = 0
        self._done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.cycle = self.rng.choice([45, 50, 55, 60])
        lo = int(self.cycle * 0.2) // 5 + 1
        hi = int(self.cycle * 0.35) // 5
        self.green_len = 5 * self.rng.randint(lo, hi)
        self.unknown_ids = [2, 3, 4]
        self.targets = {i: self.rng.randint(0, self.cycle - 1) for i in self.unknown_ids}
        self.locked = {}
        self.steps = 0
        self._done = False

        obs = (
            f"GREEN WAVE TIMING: Avenue signals 1-4, cycle length {self.cycle}s, "
            f"each with a {self.green_len}s green window. Signal 1 is the fixed "
            "reference (offset 0) and a platoon leaves it at the start of every "
            "cycle. For signals 2, 3 and 4 you must set a window-start offset "
            f"(integer 0..{self.cycle - 1}) so the platoon's arrival lands inside "
            "that signal's green window; each signal's correct offset is unknown "
            "and must be discovered by probing.\n"
            "ACTIONS: 'TEST <id> <offset>' sends a trial pulse and reports a "
            "rounded timing correction without committing anything; "
            "'LOCK <id> <offset>' commits your final offset for that signal "
            "(one lock per signal, scored immediately, further actions on a "
            f"locked signal have no effect). You have {self.MAX_STEPS} steps "
            "total across all three signals. Reward is 1/3 for each signal "
            "locked within tolerance, total 1.0 for a perfect run."
        )
        return obs, {"cycle": self.cycle, "green_len": self.green_len, "steps_left": self.MAX_STEPS}

    def _circ_diff(self, a, b):
        half = self.cycle // 2
        return (a - b + half) % self.cycle - half

    def step(self, action):
        if self._done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        steps_left = max(self.MAX_STEPS - self.steps, 0)
        parts = (action or "").strip().split()

        def finish(obs, reward):
            terminated = len(self.locked) == len(self.unknown_ids)
            truncated = (not terminated) and self.steps >= self.MAX_STEPS
            self._done = terminated or truncated
            info = {"steps_left": steps_left, "locked": sorted(self.locked)}
            return obs, reward, terminated, truncated, info

        if len(parts) != 3 or parts[0].upper() not in ("TEST", "LOCK"):
            return finish(
                "Malformed action. Use 'TEST <id> <offset>' or 'LOCK <id> <offset>' "
                f"with id in 2-4 and offset in 0-{self.cycle - 1}. ({steps_left} steps left.)",
                0.0,
            )

        cmd = parts[0].upper()
        try:
            sig_id = int(parts[1])
            offset = int(parts[2])
        except ValueError:
            return finish(
                f"Malformed action: id and offset must be integers. ({steps_left} steps left.)",
                0.0,
            )

        if sig_id not in self.unknown_ids or not (0 <= offset < self.cycle):
            return finish(
                f"Invalid target: id must be in 2-4 and offset in 0-{self.cycle - 1}. "
                f"({steps_left} steps left.)",
                0.0,
            )

        if sig_id in self.locked:
            return finish(
                f"Signal {sig_id} is already locked at offset {self.locked[sig_id]}; "
                f"this action had no effect. ({steps_left} steps left.)",
                0.0,
            )

        diff = self._circ_diff(offset, self.targets[sig_id])

        if cmd == "TEST":
            rounded = int(round(diff / 5.0)) * 5
            if rounded == 0:
                msg = f"Signal {sig_id}: pulse landed inside the green window (near-target)."
            elif rounded > 0:
                msg = (
                    f"Signal {sig_id}: pulse arrived about {rounded}s too LATE for the "
                    "window (try an earlier offset, roughly that much lower)."
                )
            else:
                msg = (
                    f"Signal {sig_id}: pulse arrived about {-rounded}s too EARLY for the "
                    "window (try a later offset, roughly that much higher)."
                )
            return finish(msg + f" ({steps_left} steps left.)", 0.0)

        # LOCK
        self.locked[sig_id] = offset
        if abs(diff) <= self.TOLERANCE:
            r = 1.0 / len(self.unknown_ids)
            msg = f"Signal {sig_id} LOCKED at {offset}: on time, platoon flows through."
            return finish(msg + f" ({steps_left} steps left.)", r)
        msg = f"Signal {sig_id} LOCKED at {offset}: platoon stops here, timing missed."
        return finish(msg + f" ({steps_left} steps left.)", 0.0)
