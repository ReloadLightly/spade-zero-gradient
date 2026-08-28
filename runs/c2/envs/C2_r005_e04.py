import random


class DriftingClockEnv:
    def __init__(self):
        self._rng = None
        self._B = None
        self._S1 = None
        self._S2 = None
        self._T = None
        self._H_T = None
        self._steps = 0
        self._reads = 0
        self._max_reads = 4
        self._max_readable_t = 6
        self._step_limit = 10
        self._done = False

    def _hour_at(self, t):
        if t <= self._B:
            return (t * self._S1) % 12
        return (self._B * self._S1 + (t - self._B) * self._S2) % 12

    def reset(self, seed=None):
        self._rng = random.Random(seed)
        self._B = self._rng.choice([3, 4, 5])
        self._S1 = self._rng.randint(1, 11)
        self._S2 = self._rng.randint(1, 11)
        while self._S2 == self._S1:
            self._S2 = self._rng.randint(1, 11)
        self._T = self._B + 4
        self._H_T = self._hour_at(self._T)
        self._steps = 0
        self._reads = 0
        self._done = False

        obs = (
            "A broken clock started at 12 (hour value 0) and has been ticking. "
            "For some number of ticks it drifted by one hidden hourly rate each tick, "
            "then permanently switched to a second, different hidden rate for all later ticks. "
            "Both rates are integers from 1 to 11 (mod 12), and the switch happens after "
            "tick 3, 4, or 5.\n\n"
            "Your goal: report both hidden rates and the exact hour value the clock shows "
            f"at tick {self._T} (which you cannot query directly).\n\n"
            "Action format (exactly one per turn):\n"
            "  'READ t'  -- query the displayed hour at tick t (1 <= t <= 6). "
            f"You have {self._max_reads} probes total.\n"
            "  'GUESS s1 s2 h'  -- submit your guess for the first rate (s1), the second "
            f"rate (s2), and the hour shown at tick {self._T} (h). Ends the episode.\n\n"
            f"You have {self._step_limit} actions total. GUESS may be submitted at any time."
        )
        return obs, {}

    def step(self, action):
        if self._done:
            return "Episode already finished.", 0.0, True, False, {}

        self._steps += 1
        text = (action or "").strip()
        parts = text.split()

        if not parts:
            obs = "Empty action. Use 'READ t' or 'GUESS s1 s2 h'."
            return obs, 0.0, False, self._check_truncate(), {}

        cmd = parts[0].upper()

        if cmd == "READ":
            if len(parts) != 2:
                obs = "Malformed READ. Use 'READ t' with a single integer tick."
                return obs, 0.0, False, self._check_truncate(), {}
            try:
                t = int(parts[1])
            except ValueError:
                obs = "Malformed READ. The tick must be an integer."
                return obs, 0.0, False, self._check_truncate(), {}
            if t < 1 or t > self._max_readable_t:
                obs = f"Tick out of range. READ accepts t from 1 to {self._max_readable_t}."
                return obs, 0.0, False, self._check_truncate(), {}
            if self._reads >= self._max_reads:
                obs = "No probes remaining. You must submit GUESS s1 s2 h."
                return obs, 0.0, False, self._check_truncate(), {}
            self._reads += 1
            reading = self._hour_at(t)
            remaining = self._max_reads - self._reads
            obs = (
                f"At tick {t}, the clock shows hour {reading}. "
                f"Probes remaining: {remaining}."
            )
            return obs, 0.0, False, self._check_truncate(), {}

        if cmd == "GUESS":
            if len(parts) != 4:
                obs = "Malformed GUESS. Use 'GUESS s1 s2 h' with three integers."
                return obs, 0.0, False, self._check_truncate(), {}
            try:
                g_s1 = int(parts[1]) % 12
                g_s2 = int(parts[2]) % 12
                g_h = int(parts[3]) % 12
            except ValueError:
                obs = "Malformed GUESS. All three values must be integers."
                return obs, 0.0, False, self._check_truncate(), {}

            reward = 0.0
            if g_s1 == self._S1:
                reward += 0.3
            if g_s2 == self._S2:
                reward += 0.3
            if g_h == self._H_T:
                reward += 0.4

            self._done = True
            obs = (
                f"Final: first rate {'correct' if g_s1 == self._S1 else 'incorrect'}, "
                f"second rate {'correct' if g_s2 == self._S2 else 'incorrect'}, "
                f"tick-{self._T} hour {'correct' if g_h == self._H_T else 'incorrect'}. "
                f"Reward earned: {reward:.1f}."
            )
            return obs, reward, True, False, {}

        obs = "Unrecognized action. Use 'READ t' or 'GUESS s1 s2 h'."
        return obs, 0.0, False, self._check_truncate(), {}

    def _check_truncate(self):
        if self._steps >= self._step_limit and not self._done:
            self._done = True
            return True
        return False
