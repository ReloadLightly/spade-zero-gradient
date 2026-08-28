import random


class TrafficGreenWaveEnv:
    CANDIDATES = [9, 10, 11, 12, 13]
    DISTANCES = {"B": 300, "C": 460, "D": 620}
    CYCLE = 60
    GREEN = 22
    MAX_STEPS = 10
    MAX_PROBES = 2

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.v = self.rng.choice(self.CANDIDATES)
        self.steps = 0
        self.probes_used = 0
        self.probed_lights = []
        self.candidate_set = set(self.CANDIDATES)
        self.m1 = False
        self.m2 = False
        self.m3 = False
        obs = (
            "Avenue timing task. Four intersections lie along the avenue: A (start, "
            "reference, offset fixed at 0), then B, C, D further down the avenue. Each "
            "signal runs a 60-second cycle with a 22-second green phase beginning at "
            "that signal's offset (an integer 0-59). A platoon departs A at the start "
            "of A's green (t=0) and travels the whole avenue at one constant but "
            "UNKNOWN true speed.\n"
            "GOAL: choose offsets for B, C, and D so the platoon arrives during green "
            "at as many of them as possible.\n"
            "Distances from A: B=300m, C=460m, D=620m.\n"
            "You may run up to 2 timed test passes before committing. Each reports the "
            "travel time from A to the chosen intersection, rounded to the nearest 5 "
            "seconds:\n"
            "  TIME <B|C|D>\n"
            "When ready, commit final offsets (this ends the episode):\n"
            "  SET <offset_B> <offset_C> <offset_D>   (each an integer 0-59)\n"
            "You have at most 10 total actions."
        )
        return obs, {}

    def _in_range(self, t, a, b):
        if a <= b:
            return a <= t < b
        return t >= a or t < b

    def _score_light(self, arrival, offset):
        lo = offset % self.CYCLE
        hi = (offset + self.GREEN) % self.CYCLE
        if self._in_range(arrival, lo, hi):
            return self.GREEN, "GREEN (full credit)"
        band_lo = (offset - 5) % self.CYCLE
        band_hi = (offset + self.GREEN + 5) % self.CYCLE
        if self._in_range(arrival, band_lo, band_hi):
            return self.GREEN / 2.0, "NEAR-GREEN (partial credit)"
        return 0.0, "RED (no credit)"

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        info = {}
        text = (action or "").strip().upper()
        parts = text.split()

        if len(parts) == 2 and parts[0] == "TIME" and parts[1] in self.DISTANCES:
            if self.probes_used >= self.MAX_PROBES:
                obs = (
                    f"Probe budget exhausted ({self.probes_used}/{self.MAX_PROBES} "
                    "used). Submit SET <offset_B> <offset_C> <offset_D> to commit."
                )
            else:
                light = parts[1]
                d = self.DISTANCES[light]
                true_time = d / self.v
                reading = round(true_time / 5.0) * 5
                self.probes_used += 1
                self.probed_lights.append(light)
                self.candidate_set = {
                    cand for cand in self.candidate_set
                    if round((self.DISTANCES[light] / cand) / 5.0) * 5 == reading
                }
                if not self.m1:
                    reward += 0.10
                    self.m1 = True
                elif not self.m2 and len(set(self.probed_lights)) >= 2:
                    reward += 0.10
                    self.m2 = True
                if not self.m3 and len(self.candidate_set) <= 3:
                    reward += 0.10
                    self.m3 = True
                obs = (
                    f"Test pass A->{light} ({d} m): elapsed ~{reading} s (rounded to "
                    f"nearest 5s). Probes used: {self.probes_used}/{self.MAX_PROBES}."
                )
        elif len(parts) == 4 and parts[0] == "SET":
            try:
                ob, oc, od = int(parts[1]), int(parts[2]), int(parts[3])
                valid_ints = True
            except ValueError:
                valid_ints = False
            if not valid_ints:
                obs = "SET needs three integers 0-59, e.g. an offset for B, C, and D."
            elif not all(0 <= o < 60 for o in (ob, oc, od)):
                obs = "Offsets must each be in range 0-59."
            else:
                terminated = True
                lines = []
                total = 0.0
                for light, offset in zip(("B", "C", "D"), (ob, oc, od)):
                    arrival = (self.DISTANCES[light] / self.v) % self.CYCLE
                    score, tag = self._score_light(arrival, offset)
                    weighted = (score / self.GREEN) * (0.70 / 3.0)
                    total += weighted
                    lines.append(
                        f"{light}: arrival ~{arrival:.1f}s, green window "
                        f"[{offset}-{(offset + self.GREEN) % 60}) -> {tag}"
                    )
                reward += total
                lines.append(f"Commit score: {total:.3f}/0.700. True speed was {self.v} m/s.")
                obs = "Commit evaluated.\n" + "\n".join(lines)
        else:
            obs = (
                "Unrecognized or malformed action. Use 'TIME B', 'TIME C', 'TIME D', "
                "or 'SET <offset_B> <offset_C> <offset_D>'."
            )

        truncated = (not terminated) and self.steps >= self.MAX_STEPS
        if truncated:
            obs += " Step limit reached; episode ends without a commit."
        return obs, reward, terminated, truncated, info
