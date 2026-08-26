import random
import re


class PhantomTrainEnv:
    STATIONS = ["A", "B", "C", "D", "E"]
    STATION_NAMES = ["Ashcombe", "Bridgeport", "Calder Cross", "Drummond", "Elmswood"]
    SPEEDS = [5, 7, 9, 11, 13]
    MAX_STEPS = 10

    def _fmt(self, t):
        return "%02d:%02d" % ((t // 60) % 24, t % 60)

    def _pred(self, cand, k):
        return cand["dep"] + (k - cand["origin"]) * cand["speed"]

    def _generate(self):
        while True:
            cands = []
            for _ in range(6):
                o = self.rng.randint(0, 3)
                d = self.rng.randint(o + 1, 4)
                speed = self.rng.choice(self.SPEEDS)
                dep = self.rng.randint(300, 419)
                cands.append({"origin": o, "dest": d, "speed": speed, "dep": dep})
            phantom = self.rng.randint(0, 5)

            def conflict():
                pc = cands[phantom]
                for k in range(pc["origin"], pc["dest"] + 1):
                    pt = self._pred(pc, k)
                    for idx, c in enumerate(cands):
                        if idx == phantom:
                            continue
                        if c["origin"] <= k <= c["dest"] and self._pred(c, k) == pt:
                            return True
                return False

            guard = 0
            while conflict() and guard < 500:
                cands[phantom]["dep"] += 1
                guard += 1
            if not conflict():
                self.candidates = cands
                self.phantom = phantom
                return

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.steps = 0
        self.done = False
        self.queried = set()
        self._generate()

        pc = self.candidates[self.phantom]
        route = list(range(pc["origin"], pc["dest"] + 1))
        n = len(route)
        base = 0.5 / n
        rewards = [base] * n
        rewards[-1] = 0.5 - sum(rewards[:-1])
        self.phantom_reward = dict(zip(route, rewards))

        lines = []
        lines.append("MISSING TRAIN INVESTIGATION")
        lines.append("Line order (west to east): " + " - ".join(
            "%s(%s)" % (s, n2) for s, n2 in zip(self.STATIONS, self.STATION_NAMES)))
        lines.append("")
        lines.append("An old paper timetable lists 6 candidate trains. Exactly ONE of")
        lines.append("them never actually ran (a phantom working); the other 5 ran")
        lines.append("exactly as claimed. A real train's predicted arrival time WILL")
        lines.append("appear in the true station log for every station on its route;")
        lines.append("the phantom's predicted time will appear at NONE of them.")
        lines.append("")
        lines.append("Predicted arrival at station k = departure + (k - origin) * speed")
        lines.append("(k counted in station positions along the route, origin..dest).")
        lines.append("")
        lines.append("Candidates:")
        lines.append("ID  Origin  Dest  Departure  Speed(min/segment)")
        for i, c in enumerate(self.candidates):
            lines.append("%d   %s       %s     %s      %d" % (
                i + 1, self.STATIONS[c["origin"]], self.STATIONS[c["dest"]],
                self._fmt(c["dep"]), c["speed"]))
        lines.append("")
        lines.append("GOAL: identify the phantom candidate's ID.")
        lines.append("ACTIONS:")
        lines.append("  LOG <station letter>   - reveal true recorded arrival times there")
        lines.append("  GUESS <candidate id>   - final answer (ends the episode)")
        lines.append("You have %d steps total." % self.MAX_STEPS)

        return "\n".join(lines), {}

    def _station_log(self, k):
        times = set()
        for idx, c in enumerate(self.candidates):
            if idx == self.phantom:
                continue
            if c["origin"] <= k <= c["dest"]:
                times.add(self._pred(c, k))
        return sorted(times)

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()

        m_log = re.match(r"^LOG\s+([A-Za-z])$", action, re.IGNORECASE)
        m_guess = re.match(r"^GUESS\s+(\d+)$", action, re.IGNORECASE)

        reward = 0.0
        terminated = False
        obs = ""

        if m_log:
            letter = m_log.group(1).upper()
            if letter not in self.STATIONS:
                obs = "Unknown station '%s'. Valid stations: %s." % (
                    letter, ", ".join(self.STATIONS))
            else:
                k = self.STATIONS.index(letter)
                log = self._station_log(k)
                if k not in self.queried:
                    self.queried.add(k)
                    reward = self.phantom_reward.get(k, 0.0)
                if log:
                    obs = "Station %s (%s) true arrival log: %s" % (
                        letter, self.STATION_NAMES[k],
                        ", ".join(self._fmt(t) for t in log))
                else:
                    obs = "Station %s (%s): no arrivals recorded." % (
                        letter, self.STATION_NAMES[k])
        elif m_guess:
            cid = int(m_guess.group(1))
            if not (1 <= cid <= 6):
                obs = "Invalid candidate id '%d'. Choose an ID from 1 to 6." % cid
            else:
                terminated = True
                self.done = True
                if cid - 1 == self.phantom:
                    reward = 0.5
                    obs = "Correct. Candidate %d never ran; it was the phantom working." % cid
                else:
                    reward = 0.0
                    obs = "Incorrect. Candidate %d did run. The phantom was candidate %d." % (
                        cid, self.phantom + 1)
        else:
            obs = ("Malformed action. Use 'LOG <station letter>' or "
                   "'GUESS <candidate id>'.")

        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += " Step limit reached; investigation closed without a guess."

        return obs, reward, terminated, truncated, {"steps_remaining": max(0, self.MAX_STEPS - self.steps)}
