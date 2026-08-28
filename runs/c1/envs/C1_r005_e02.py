import itertools
import random


class LoadingDockRelayEnv:
    TRUCKS = ["A", "B", "C", "D"]
    PROBE_BUDGET = 6
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.p = {t: self.rng.randint(3, 9) for t in self.TRUCKS}
        self.d = {t: self.rng.randint(6, 24) for t in self.TRUCKS}
        self.r = {t: self.rng.randint(1, 5) for t in self.TRUCKS}
        self.c = {}
        for i in self.TRUCKS:
            for j in self.TRUCKS:
                if i != j:
                    self.c[(i, j)] = self.rng.randint(0, 6)
        self.probes_used = 0
        self.steps = 0
        self.done = False
        return self._intro(), {}

    def _intro(self):
        lines = [
            "You run a single loading dock. Four trucks - A, B, C, D - must",
            "each be dispatched exactly once, one at a time, in an order you",
            "choose. Below are each truck's processing time (minutes to",
            "load), deadline (minutes after the dock opens), and late-",
            "penalty rate (cost per minute a truck finishes past its",
            "deadline).",
            "",
            "Truck  processing  deadline  late_penalty_rate",
        ]
        for t in self.TRUCKS:
            lines.append(f"  {t}        {self.p[t]:>2}          {self.d[t]:>2}         {self.r[t]:>2}")
        lines += [
            "",
            "Hidden: switching the dock directly from one truck to another",
            "costs an extra CHANGEOVER delay (minutes added before the next",
            "truck can start loading) that depends on the exact pair and",
            "direction. You do not know any changeover value until you",
            "probe it. The first truck in your order has no changeover.",
            "",
            "Goal: choose a dispatch ORDER containing all four trucks,",
            "each exactly once, minimizing total late penalty = sum over",
            "trucks of rate * (completion_time - deadline), counted only",
            "for trucks that finish after their own deadline.",
            "",
            "Actions (send exactly one per turn):",
            "  PROBE <X> <Y>   reveal the changeover delay incurred if Y is",
            "                  dispatched immediately after X (uses 1 of",
            f"                  your {self.PROBE_BUDGET} probes)",
            "  SUBMIT <order>  e.g. 'SUBMIT B A D C' - commit your final",
            "                  dispatch order. Irreversible; ends the episode.",
            "",
            f"You have at most {self.MAX_STEPS} total actions (probes plus the",
            "final SUBMIT combined).",
        ]
        return "\n".join(lines)

    def _score(self, order):
        t = 0
        penalty = 0
        prev = None
        for truck in order:
            if prev is not None:
                t += self.c[(prev, truck)]
            t += self.p[truck]
            if t > self.d[truck]:
                penalty += self.r[truck] * (t - self.d[truck])
            prev = truck
        return penalty

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        text = (action or "").strip()
        parts = text.split()
        cmd = parts[0].upper() if parts else ""

        if cmd == "PROBE":
            obs, reward = self._do_probe(parts)
        elif cmd == "SUBMIT":
            obs, reward = self._do_submit(parts)
        else:
            obs = (
                f"Unrecognized action '{text}'. Use 'PROBE <X> <Y>' or "
                "'SUBMIT <order of all four trucks>'."
            )
            reward = 0.0

        terminated = self.done
        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            obs += "\n\nStep limit reached without a SUBMIT - episode ends with no schedule committed."

        return obs, reward, terminated, truncated, {}

    def _do_probe(self, parts):
        if len(parts) != 3:
            return (
                "Invalid PROBE - format is 'PROBE <X> <Y>' with two "
                "different trucks from A, B, C, D.",
                0.0,
            )
        x, y = parts[1].upper(), parts[2].upper()
        if x not in self.TRUCKS or y not in self.TRUCKS or x == y:
            return (
                "Invalid PROBE - both arguments must be different trucks "
                "from A, B, C, D.",
                0.0,
            )
        if self.probes_used >= self.PROBE_BUDGET:
            return (
                f"Probe budget exhausted ({self.probes_used}/{self.PROBE_BUDGET} "
                "used) - no more PROBE actions allowed; you may still SUBMIT.",
                0.0,
            )
        self.probes_used += 1
        delay = self.c[(x, y)]
        return (
            f"Changeover delay from {x} to {y} is {delay} minutes. "
            f"Probes used: {self.probes_used}/{self.PROBE_BUDGET}. "
            f"Steps used: {self.steps}/{self.MAX_STEPS}.",
            0.0,
        )

    def _do_submit(self, parts):
        order = [p.upper() for p in parts[1:]]
        if sorted(order) != sorted(self.TRUCKS):
            return (
                "Invalid SUBMIT - must list all four trucks A, B, C, D "
                "exactly once, e.g. 'SUBMIT A B C D'.",
                0.0,
            )

        achieved = self._score(order)
        baseline = self._score(sorted(self.TRUCKS))
        optimal = min(self._score(list(o)) for o in itertools.permutations(self.TRUCKS))

        improved = 0.4 if achieved <= baseline else 0.0
        if achieved <= baseline:
            if baseline == optimal:
                closeness = 1.0 if achieved == optimal else 0.0
            else:
                closeness = (baseline - achieved) / (baseline - optimal)
                closeness = max(0.0, min(1.0, closeness))
        else:
            closeness = 0.0
        reward = round(improved + 0.6 * closeness, 4)
        reward = max(0.0, min(1.0, reward))

        self.done = True
        obs = (
            f"SUBMIT accepted: order {' '.join(order)}.\n"
            f"Total late penalty achieved: {achieved}.\n"
            f"Naive alphabetical-order penalty (baseline): {baseline}.\n"
            f"Best possible penalty (optimal): {optimal}.\n"
            f"Reward: {reward}."
        )
        return obs, reward
