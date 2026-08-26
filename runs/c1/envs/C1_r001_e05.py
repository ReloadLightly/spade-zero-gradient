import random
import re


class GreenWaveBoulevardEnv:
    """Optimize a single platoon departure time to maximize the number of
    synchronized traffic signals hit on green along a 5-light avenue."""

    NUM_LIGHTS = 5
    CYCLE = 20
    GREEN_LEN = 10
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.travel_cum = []
        self.offsets = []
        self.steps = 0
        self.done = False
        self.best_clean = 0

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.steps = 0
        self.done = False
        self.best_clean = 0

        gaps = [self.rng.randint(2, 7) for _ in range(self.NUM_LIGHTS)]
        cum = []
        total = 0
        for g in gaps:
            total += g
            cum.append(total)
        self.travel_cum = cum

        target = self.rng.randint(0, self.CYCLE - 1)
        offsets = []
        for i in range(self.NUM_LIGHTS):
            arrival = (target + self.travel_cum[i]) % self.CYCLE
            jitter = self.rng.randint(0, self.GREEN_LEN - 1)
            offsets.append((arrival - jitter) % self.CYCLE)
        self.offsets = offsets

        dist_str = ", ".join(
            f"L{i + 1} at {self.travel_cum[i]} units" for i in range(self.NUM_LIGHTS)
        )
        obs = (
            "AVENUE SIGNAL SYNC\n"
            f"Five traffic lights (L1..L5) line an avenue. Every light shares the "
            f"same {self.CYCLE}-unit cycle and stays green for {self.GREEN_LEN} "
            "consecutive units of every cycle, then red for the rest -- but each "
            "light's green WINDOW start (its phase offset) is hidden and different "
            "per light.\n"
            f"Distances from the avenue start, in travel-time units: {dist_str}.\n"
            "Your job: pick one departure time t (an integer, 0-19, cyclic) for a "
            "platoon leaving the avenue start, so that it arrives at each light "
            "during that light's green window. A platoon leaving at time t reaches "
            f"light i at time (t + distance_i) mod {self.CYCLE}.\n"
            "Actions:\n"
            "  TEST <t>   -- simulate a departure at time t (0-19); costs a step, "
            "no reward, reports how many of the 5 lights would be hit on green.\n"
            "  SUBMIT <t> -- commit to departure time t; ends the episode; reward "
            "= (lights hit green) / 5.\n"
            f"You have {self.MAX_STEPS} total actions (TEST + SUBMIT combined). "
            "Choose wisely."
        )
        return obs, {}

    def _clean_count(self, t):
        clean = 0
        for i in range(self.NUM_LIGHTS):
            arrival = (t + self.travel_cum[i]) % self.CYCLE
            start = self.offsets[i]
            delta = (arrival - start) % self.CYCLE
            if delta < self.GREEN_LEN:
                clean += 1
        return clean

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        text = (action or "").strip()
        m = re.match(r'^(TEST|SUBMIT)\s+([+-]?\d+)$', text, re.IGNORECASE)

        if not m:
            obs = (
                "Malformed action. Use 'TEST <t>' or 'SUBMIT <t>' with an "
                "integer t between 0 and 19."
            )
            truncated = self.steps >= self.MAX_STEPS
            if truncated:
                self.done = True
            return obs, 0.0, False, truncated, {}

        verb = m.group(1).upper()
        t = int(m.group(2))

        if not (0 <= t <= self.CYCLE - 1):
            obs = f"t must be between 0 and {self.CYCLE - 1}. No action taken."
            truncated = self.steps >= self.MAX_STEPS
            if truncated:
                self.done = True
            return obs, 0.0, False, truncated, {}

        clean = self._clean_count(t)

        if verb == "TEST":
            self.best_clean = max(self.best_clean, clean)
            obs = (
                f"t={t}: {clean}/5 lights hit green (best so far: "
                f"{self.best_clean}/5)."
            )
            truncated = self.steps >= self.MAX_STEPS
            if truncated:
                self.done = True
                obs += " Step budget exhausted without a SUBMIT -- episode over."
            return obs, 0.0, False, truncated, {"clean_count": clean}

        # SUBMIT
        reward = clean / self.NUM_LIGHTS
        self.done = True
        obs = f"SUBMIT t={t}: {clean}/5 lights hit green. Final reward {reward:.2f}."
        return obs, reward, True, False, {"clean_count": clean}
