import random


class PortfolioBudgetEnv:
    NUM_PROJECTS = 6
    MAX_APPRAISALS = 4
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.labels = []
        self.costs = []
        self.values = []
        self.ranges = []
        self.budget = 0
        self.appraised = {}
        self.appraisals_left = self.MAX_APPRAISALS
        self.step_count = 0
        self.done = False
        self._optimal_value = 0

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        n = self.NUM_PROJECTS
        self.labels = list("ABCDEF")[:n]

        self.costs = [self.rng.randint(2, 9) for _ in range(n)]
        self.values = []
        for c in self.costs:
            base = c * self.rng.randint(1, 3)
            noise = self.rng.randint(-3, 4)
            self.values.append(max(1, base + noise))

        self.ranges = []
        for v in self.values:
            spread = self.rng.randint(3, 6)
            lo = max(1, v - spread)
            hi = v + self.rng.randint(0, spread)
            self.ranges.append((lo, hi))

        total_cost = sum(self.costs)
        budget = int(total_cost * self.rng.uniform(0.45, 0.6))
        self.budget = max(budget, min(self.costs) + 2)

        self.appraised = {}
        self.appraisals_left = self.MAX_APPRAISALS
        self.step_count = 0
        self.done = False
        self._optimal_value = self._compute_optimal()

        obs = self._render_intro()
        info = {"budget": self.budget, "costs": dict(zip(self.labels, self.costs))}
        return obs, info

    def _compute_optimal(self):
        n = len(self.costs)
        B = self.budget
        dp = [0] * (B + 1)
        for i in range(n):
            c, v = self.costs[i], self.values[i]
            for b in range(B, c - 1, -1):
                cand = dp[b - c] + v
                if cand > dp[b]:
                    dp[b] = cand
        return max(dp)

    def _render_intro(self):
        lines = [
            "PORTFOLIO SELECTION UNDER BUDGET",
            f"Choose a subset of projects ({', '.join(self.labels)}) whose total COST "
            f"does not exceed budget {self.budget}, maximizing total hidden VALUE.",
            "Each project's COST is public now. Its exact VALUE is hidden until "
            "appraised; only a coarse range is given up front.",
            f"You may APPRAISE at most {self.MAX_APPRAISALS} projects total "
            "(each appraisal is one action and reveals the exact value).",
            "Actions:",
            "  APPRAISE <label>   -- reveal one project's exact value (uses one appraisal)",
            "  SELECT <labels>    -- e.g. SELECT A,C,D -- final choice, ends the episode",
            f"You have at most {self.MAX_STEPS} actions total; if you run out without "
            "selecting, the episode truncates with 0 reward.",
            "Projects:",
        ]
        for lbl, c, (lo, hi) in zip(self.labels, self.costs, self.ranges):
            lines.append(f"  {lbl}: cost={c}, estimated value in [{lo}, {hi}]")
        return "\n".join(lines)

    def _status_line(self):
        known = ", ".join(f"{l}={v}" for l, v in self.appraised.items()) or "none"
        return (f"Appraisals left: {self.appraisals_left}. Steps left: "
                f"{self.MAX_STEPS - self.step_count}. Known values: {known}.")

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()
        upper = text.upper()

        if upper.startswith("APPRAISE"):
            arg = text[len("APPRAISE"):].strip().upper()
            reward, obs, terminated = self._do_appraise(arg)
        elif upper.startswith("SELECT"):
            arg = text[len("SELECT"):].strip()
            reward, obs, terminated = self._do_select(arg)
        else:
            reward = 0.0
            terminated = False
            obs = ("Unrecognized action. Use 'APPRAISE <label>' or "
                   "'SELECT <label,label,...>'. " + self._status_line())

        truncated = False
        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated = True
            obs += " Step limit reached; episode truncated."

        if terminated or truncated:
            self.done = True

        return obs, reward, terminated, truncated, {}

    def _do_appraise(self, label):
        if label not in self.labels:
            return 0.0, f"Unknown label '{label}'. Valid labels: {', '.join(self.labels)}. " + self._status_line(), False
        if label in self.appraised:
            return 0.0, f"{label} was already appraised (value={self.appraised[label]}). " + self._status_line(), False
        if self.appraisals_left <= 0:
            return 0.0, "No appraisals remaining; you must SELECT now. " + self._status_line(), False

        idx = self.labels.index(label)
        v = self.values[idx]
        self.appraised[label] = v
        self.appraisals_left -= 1
        obs = f"Appraised {label}: exact value = {v}. " + self._status_line()
        return 0.0, obs, False

    def _do_select(self, arg):
        raw = [p.strip().upper() for p in arg.split(",") if p.strip()]
        if not raw:
            return 0.0, "SELECT requires at least one label, e.g. SELECT A,C. " + self._status_line(), False
        if len(set(raw)) != len(raw):
            return 0.0, "Duplicate label in SELECT list; each project may appear once. " + self._status_line(), False
        for lbl in raw:
            if lbl not in self.labels:
                return 0.0, f"Unknown label '{lbl}' in SELECT. Valid labels: {', '.join(self.labels)}. " + self._status_line(), False

        idxs = [self.labels.index(l) for l in raw]
        total_cost = sum(self.costs[i] for i in idxs)
        total_value = sum(self.values[i] for i in idxs)

        if total_cost > self.budget:
            obs = (f"Infeasible: selected cost {total_cost} exceeds budget {self.budget}. "
                   "Episode over.")
            return 0.0, obs, True

        feasibility_reward = 0.2
        if self._optimal_value > 0:
            quality_reward = 0.8 * (total_value / self._optimal_value)
        else:
            quality_reward = 0.8
        reward = min(1.0, feasibility_reward + quality_reward)

        obs = (f"SELECT accepted: {','.join(raw)} -- cost={total_cost}/{self.budget}, "
               f"value={total_value} (best possible was {self._optimal_value}). "
               f"Reward={reward:.3f}. Episode over.")
        return reward, obs, True
