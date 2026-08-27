import random
import re


class AvenueGreenWaveEnv:
    def __init__(self):
        self.rng = None
        self.C = 20
        self.G = 5
        self.V_MIN = 4
        self.V_MAX = 15
        self.max_steps = 10
        self.p = []
        self.offsets = []
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        gaps = [self.rng.randint(60, 140) for _ in range(4)]
        self.p = []
        cum = 0
        for g in gaps:
            cum += g
            self.p.append(cum)
        v_star = self.rng.randint(self.V_MIN, self.V_MAX)
        self.offsets = []
        for pi in self.p:
            t = pi / v_star
            o = (t - self.G / 2.0) % self.C
            self.offsets.append(o)
        self.steps = 0
        self.done = False

        dist_str = ", ".join(f"Light {i+1}: {pi}m" for i, pi in enumerate(self.p))
        obs = (
            "AVENUE GREEN-WAVE\n"
            "You drive from the west end of the avenue at t=0s. Four traffic lights "
            f"lie ahead at known distances: {dist_str}. Every light repeats on a cycle "
            f"of {self.C}s and is green for {self.G}s of each cycle, but each light's "
            "OFFSET (when in the cycle its green begins) is hidden and independent.\n"
            f"Choose one constant speed (an integer from {self.V_MIN} to {self.V_MAX} "
            "m/s) for your whole drive. Your goal is to pick the speed that gets you "
            "through as many of the 4 lights on green as possible, then commit.\n"
            "Actions:\n"
            "  TEST <v>   - probe speed v (integer m/s): reports, for each light, "
            "whether you'd hit GREEN, or if RED, how many seconds earlier or later "
            "you'd need to arrive to catch that light's green. Costs a step, no reward.\n"
            "  SUBMIT <v> - commit to speed v permanently, ending the episode. Reward "
            "is 0.25 per light you hit on green (up to 1.0 for all 4).\n"
            f"You have {self.max_steps} steps total (TEST and SUBMIT both count)."
        )
        return obs, {}

    def _evaluate(self, v):
        results = []
        green_count = 0
        for pi, oi in zip(self.p, self.offsets):
            t = pi / v
            phase = (t - oi) % self.C
            if phase < self.G:
                results.append((True, t, None))
                green_count += 1
            else:
                fwd = self.C - phase
                bwd = phase - self.G
                results.append((False, t, (fwd, bwd)))
        return green_count, results

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        m = re.match(r"^\s*(TEST|SUBMIT)\s+(\d+)\s*$", action or "", re.IGNORECASE)
        if not m:
            obs = (
                "Malformed action. Use 'TEST <v>' or 'SUBMIT <v>' with an integer "
                f"v between {self.V_MIN} and {self.V_MAX}."
            )
            return obs, 0.0, False, self.steps >= self.max_steps, {}

        kind = m.group(1).upper()
        v = int(m.group(2))
        if v < self.V_MIN or v > self.V_MAX:
            obs = f"Speed {v} is out of range; choose an integer from {self.V_MIN} to {self.V_MAX}."
            return obs, 0.0, False, self.steps >= self.max_steps, {}

        green_count, results = self._evaluate(v)

        if kind == "TEST":
            lines = []
            for i, (is_green, t, gap) in enumerate(results):
                if is_green:
                    lines.append(f"Light {i+1}: GREEN (arrive t={t:.2f}s)")
                else:
                    fwd, bwd = gap
                    if fwd <= bwd:
                        lines.append(f"Light {i+1}: RED — arrive {fwd:.2f}s LATER to catch green")
                    else:
                        lines.append(f"Light {i+1}: RED — arrive {bwd:.2f}s EARLIER to catch green")
            obs = (
                f"TEST v={v}: {green_count}/4 lights green.\n"
                + "\n".join(lines)
                + f"\nSteps used: {self.steps}/{self.max_steps}."
            )
            return obs, 0.0, False, self.steps >= self.max_steps, {}

        reward = green_count * 0.25
        self.done = True
        detail = "\n".join(
            f"Light {i+1}: {'GREEN' if r[0] else 'RED'}" for i, r in enumerate(results)
        )
        obs = (
            f"SUBMIT v={v} — final result: {green_count}/4 lights green. "
            f"Reward={reward:.2f}.\n{detail}"
        )
        return obs, reward, True, False, {}
