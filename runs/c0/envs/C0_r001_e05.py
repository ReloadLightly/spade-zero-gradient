import random
import re


class GreenWaveEnv:
    """
    Solver plays a traffic engineer choosing a single departure time from
    the start of an avenue so a platoon hits as many downstream green
    lights as possible. Each signal's phase offset is hidden and must be
    discovered via PROBE before an informed DEPART can be made.
    """

    CYCLE = 48
    GREEN = 16
    N = 4
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.offsets = []
        self.travel_times = []
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.offsets = [self.rng.randint(0, self.CYCLE - 1) for _ in range(self.N)]
        cum = 0
        self.travel_times = []
        for _ in range(self.N):
            cum += self.rng.randint(6, 15)
            self.travel_times.append(cum)
        self.steps = 0
        self.done = False

        travel_str = ", ".join(
            f"intersection {i+1}: {t}s" for i, t in enumerate(self.travel_times)
        )
        obs = (
            "GREEN WAVE COORDINATOR\n"
            f"An avenue has {self.N} signals in sequence. Every signal shares the same "
            f"{self.CYCLE}-second cycle with a {self.GREEN}-second green window; the "
            "window's start time (offset, 0-47) differs per signal and is UNKNOWN to you.\n"
            f"Travel time from your departure point to each signal (fixed): {travel_str}.\n"
            "Goal: choose one departure time t0 (0-47) so that, for as many signals as "
            "possible, your arrival time (t0 + travel_time, mod 48) falls inside that "
            "signal's green window.\n"
            "Actions (send exactly one per turn):\n"
            "  PROBE <i> <t>   - ask what signal i (1-4) shows if something arrives at "
            "time t (0-47, mod 48 applied). Reveals GREEN/RED plus timing detail.\n"
            "  TEST <t0>       - free trial run: reports GREEN/RED hit at every signal "
            "for that t0, without detail, and does not end the episode.\n"
            "  DEPART <t0>     - final, scored, ends the episode.\n"
            f"You have {self.MAX_STEPS} actions total. Reward is 0.25 per signal hit "
            "green at the moment you DEPART (max 1.0)."
        )
        return obs, {}

    def _within_green(self, arrival, offset):
        diff = (arrival - offset) % self.CYCLE
        return diff < self.GREEN

    def _hits_for(self, t0):
        hits = []
        for i in range(self.N):
            arrival = (t0 + self.travel_times[i]) % self.CYCLE
            hits.append(self._within_green(arrival, self.offsets[i]))
        return hits

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        text = (action or "").strip()

        m = re.match(r"^PROBE\s+(-?\d+)\s+(-?\d+)$", text, re.IGNORECASE)
        if m:
            i = int(m.group(1))
            t = int(m.group(2))
            if i < 1 or i > self.N:
                obs = f"Malformed PROBE: intersection must be 1-{self.N}."
                return self._maybe_truncate(obs, 0.0)
            arrival = t % self.CYCLE
            offset = self.offsets[i - 1]
            diff = (arrival - offset) % self.CYCLE
            if diff < self.GREEN:
                remaining = self.GREEN - diff
                obs = (
                    f"Intersection {i} at t={arrival}: GREEN, {remaining}s remain "
                    "before it turns red."
                )
            else:
                time_to_green = (offset - arrival) % self.CYCLE
                obs = (
                    f"Intersection {i} at t={arrival}: RED, green begins in "
                    f"{time_to_green}s."
                )
            return self._maybe_truncate(obs, 0.0)

        m = re.match(r"^TEST\s+(-?\d+)$", text, re.IGNORECASE)
        if m:
            t0 = int(m.group(1)) % self.CYCLE
            hits = self._hits_for(t0)
            parts = [f"int{idx+1}:{'GREEN' if h else 'RED'}" for idx, h in enumerate(hits)]
            obs = f"TEST t0={t0} -> " + ", ".join(parts) + f" ({sum(hits)}/{self.N} green)."
            return self._maybe_truncate(obs, 0.0)

        m = re.match(r"^DEPART\s+(-?\d+)$", text, re.IGNORECASE)
        if m:
            t0 = int(m.group(1)) % self.CYCLE
            hits = self._hits_for(t0)
            count = sum(hits)
            reward = 0.25 * count
            self.done = True
            parts = [f"int{idx+1}:{'GREEN' if h else 'RED'}" for idx, h in enumerate(hits)]
            obs = (
                f"DEPART t0={t0} -> " + ", ".join(parts) +
                f" | {count}/{self.N} signals hit green. Final reward: {reward:.2f}."
            )
            return obs, reward, True, False, {"hits": hits, "t0": t0}

        obs = (
            "Malformed action. Use 'PROBE <intersection 1-4> <time>', "
            "'TEST <t0>', or 'DEPART <t0>'."
        )
        return self._maybe_truncate(obs, 0.0)

    def _maybe_truncate(self, obs, reward):
        if self.steps >= self.MAX_STEPS:
            self.done = True
            obs = obs + f" [Step limit ({self.MAX_STEPS}) reached without DEPART — episode truncated.]"
            return obs, reward, False, True, {}
        return obs, reward, False, False, {}
