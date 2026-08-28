import random
import re


class GardenHoseRepositionEnv:
    def __init__(self):
        self.positions = []
        self.true_R = None
        self.min_k = None
        self.num_plots = 6
        self.domain_max = 30
        self.max_steps = 10
        self.max_probes = 4
        self.step_count = 0
        self.probes_used = 0
        self.done = False
        self.rng = None

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.R_candidates = [2, 3, 4, 5, 6]
        best = None
        for _ in range(300):
            positions = sorted(self.rng.sample(range(0, self.domain_max + 1), self.num_plots))
            gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            if min(gaps) < 2:
                continue
            R = self.rng.choice(self.R_candidates)
            k = self._min_positions(positions, R)
            if 2 <= k <= self.num_plots - 1:
                best = (positions, R, k)
                break
            best = best or (positions, R, k)
        self.positions, self.true_R, self.min_k = best
        self.step_count = 0
        self.probes_used = 0
        self.done = False
        return self._initial_obs(), {}

    def _min_positions(self, positions, R):
        i = 0
        n = len(positions)
        count = 0
        while i < n:
            cover_limit = positions[i] + R
            count += 1
            while i < n and positions[i] <= cover_limit:
                i += 1
        return count

    def _initial_obs(self):
        pos_str = ", ".join(str(p) for p in self.positions)
        return (
            f"GARDEN WATERING: {self.num_plots} plots sit along a hose line at fixed positions "
            f"[{pos_str}] (integer positions 0-{self.domain_max}). Placing the hose at any integer "
            f"position waters every plot within a hidden reach R of that position (R is a fixed "
            f"integer between {min(self.R_candidates)} and {max(self.R_candidates)}, constant all "
            f"episode). Goal: water ALL plots using the fewest hose repositions possible.\n"
            f"Actions:\n"
            f"  PROBE <pos> - place the hose at integer <pos> (0-{self.domain_max}) and learn how "
            f"many plots (not which) it waters from there. Limited to {self.max_probes} probes total.\n"
            f"  PLAN <p1,p2,...> - commit your final comma-separated list of hose positions. This is "
            f"your only commit and ends the episode immediately.\n"
            f"You have {self.max_steps} steps total. Reward: 0.4 if your plan waters every plot, plus "
            f"up to 0.6 more the closer your repositioning count is to the true minimum."
        )

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}
        self.step_count += 1
        action = (action or "").strip()

        m = re.match(r'^PROBE\s+(-?\d+)$', action, re.IGNORECASE)
        if m:
            if self.probes_used >= self.max_probes:
                obs = f"Probe budget exhausted ({self.max_probes} used). Submit your PLAN now."
            else:
                pos = int(m.group(1))
                if pos < 0 or pos > self.domain_max:
                    obs = f"Invalid PROBE position: must be an integer 0-{self.domain_max}."
                else:
                    self.probes_used += 1
                    count = sum(1 for p in self.positions if abs(p - pos) <= self.true_R)
                    obs = (f"PROBE {pos}: waters {count} of {self.num_plots} plots "
                            f"({self.max_probes - self.probes_used} probes left).")
            truncated = self.step_count >= self.max_steps
            if truncated:
                self.done = True
                obs += " Step limit reached without a PLAN submission."
            return obs, 0.0, False, truncated, {}

        m = re.match(r'^PLAN\s+(.+)$', action, re.IGNORECASE)
        if m:
            raw = m.group(1)
            try:
                plan = [int(x.strip()) for x in raw.split(',') if x.strip() != '']
            except ValueError:
                plan = []
            if not plan or any(p < 0 or p > self.domain_max for p in plan):
                obs = f"Invalid PLAN: give comma-separated integers within 0-{self.domain_max}."
                truncated = self.step_count >= self.max_steps
                self.done = truncated
                return obs, 0.0, False, truncated, {}

            covered = set()
            for hp in plan:
                for p in self.positions:
                    if abs(p - hp) <= self.true_R:
                        covered.add(p)
            self.done = True
            if len(covered) < self.num_plots:
                missed = self.num_plots - len(covered)
                obs = (f"PLAN failed: {missed} plot(s) never watered with the true reach "
                        f"R={self.true_R}. Episode over.")
                return obs, 0.0, True, False, {}

            used_k = len(plan)
            reward = 0.4
            if used_k <= self.min_k:
                reward += 0.6
                obs = (f"PLAN succeeds: all plots watered in {used_k} reposition(s), matching the "
                        f"true minimum ({self.min_k}) for R={self.true_R}. Optimal!")
            else:
                eff = self.min_k / used_k
                reward += 0.6 * eff
                obs = (f"PLAN succeeds: all plots watered in {used_k} reposition(s), but the true "
                        f"minimum was {self.min_k} for R={self.true_R}. Partial credit for efficiency.")
            return obs, reward, True, False, {}

        obs = "Malformed action. Use 'PROBE <pos>' or 'PLAN <p1,p2,...>'."
        truncated = self.step_count >= self.max_steps
        self.done = truncated
        return obs, 0.0, False, truncated, {}
