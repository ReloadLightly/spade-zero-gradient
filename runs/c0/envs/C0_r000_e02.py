import random
import re


class PortfolioOptimizationEnv:
    def __init__(self):
        self.n = 6
        self.max_steps = 10
        self.rng = None
        self.costs = []
        self.low = []
        self.high = []
        self.true_values = []
        self.budget = 0
        self.optimal_value = 0.0
        self.surveyed = set()
        self.survey_reward_count = 0
        self.step_count = 0
        self.terminated_flag = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.costs = [self.rng.randint(3, 9) for _ in range(self.n)]
        self.low = []
        self.high = []
        self.true_values = []
        for _ in range(self.n):
            lo = self.rng.randint(2, 6)
            hi = lo + self.rng.randint(3, 8)
            tier = self.rng.randint(0, 4)
            val = round(lo + tier * (hi - lo) / 4.0, 1)
            self.low.append(lo)
            self.high.append(hi)
            self.true_values.append(val)
        self.budget = round(sum(self.costs) * 0.55)
        if self.budget < min(self.costs):
            self.budget = min(self.costs) + self.rng.randint(1, 3)
        self.optimal_value = self._compute_optimal()
        self.surveyed = set()
        self.survey_reward_count = 0
        self.step_count = 0
        self.terminated_flag = False
        return self._render_intro(), {}

    def _compute_optimal(self):
        best = 0.0
        for mask in range(1 << self.n):
            cost = 0
            val = 0.0
            for i in range(self.n):
                if mask & (1 << i):
                    cost += self.costs[i]
                    val += self.true_values[i]
            if cost <= self.budget and val > best:
                best = val
        return best

    def _render_intro(self):
        lines = []
        lines.append("PORTFOLIO OPTIMIZATION UNDER BUDGET")
        lines.append(
            "Select a subset of the %d candidate projects to fund within a "
            "fixed budget, maximizing total realized value." % self.n
        )
        lines.append(
            "Each project's cost is known exactly. Its TRUE value is hidden "
            "-- you only see an estimated range until you survey it."
        )
        lines.append("Action format (exactly one per turn):")
        lines.append(
            "  SURVEY <id>          - reveal the exact true value of "
            "project <id> (costs one step, not budget)."
        )
        lines.append(
            "  SELECT <id,id,...>   - commit your final portfolio and end "
            "the episode (comma or space separated ids)."
        )
        lines.append(
            "You have at most %d total steps (surveys + final select "
            "combined)." % self.max_steps
        )
        lines.append("Projects:")
        for i in range(self.n):
            lines.append(
                "  %d. Project %s -- cost %d, estimated value range [%d-%d]"
                % (i + 1, chr(65 + i), self.costs[i], self.low[i], self.high[i])
            )
        lines.append("Budget: %d" % self.budget)
        lines.append("Steps used: 0/%d." % self.max_steps)
        return "\n".join(lines)

    def step(self, action):
        if self.terminated_flag:
            return "Episode already over.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()
        upper = text.upper()
        reward = 0.0
        terminated = False
        truncated = False

        if upper.startswith("SURVEY"):
            nums = re.findall(r"\d+", text)
            if len(nums) != 1:
                obs = (
                    "Malformed SURVEY action. Use exactly: SURVEY <id> "
                    "where id is 1-%d." % self.n
                )
            else:
                idx = int(nums[0]) - 1
                if idx < 0 or idx >= self.n:
                    obs = "Invalid project id. Choose 1-%d." % self.n
                else:
                    first_time = idx not in self.surveyed
                    self.surveyed.add(idx)
                    if first_time and self.survey_reward_count < 3:
                        reward = 0.1
                        self.survey_reward_count += 1
                    obs = (
                        "Survey result -- Project %s: true value = %.1f "
                        "(cost %d)." % (chr(65 + idx), self.true_values[idx], self.costs[idx])
                    )
        elif upper.startswith("SELECT"):
            nums = re.findall(r"\d+", text)
            ids = sorted(set(int(x) - 1 for x in nums))
            invalid = [x for x in ids if x < 0 or x >= self.n]
            if not ids or invalid:
                obs = (
                    "Malformed SELECT action. Use: SELECT <id,id,...> "
                    "with ids 1-%d." % self.n
                )
            else:
                total_cost = sum(self.costs[i] for i in ids)
                if total_cost > self.budget:
                    obs = (
                        "Portfolio rejected: total cost %d exceeds budget "
                        "%d. Episode over." % (total_cost, self.budget)
                    )
                    terminated = True
                else:
                    achieved = sum(self.true_values[i] for i in ids)
                    ratio = (
                        achieved / self.optimal_value
                        if self.optimal_value > 0
                        else 1.0
                    )
                    ratio = min(ratio, 1.0)
                    reward = 0.1 + 0.6 * ratio
                    obs = (
                        "Portfolio accepted. Cost %d/%d. Achieved value %.1f "
                        "(optimal %.1f, %.0f%% of optimal). Episode over."
                        % (total_cost, self.budget, achieved, self.optimal_value, ratio * 100)
                    )
                    terminated = True
        else:
            obs = (
                "Unrecognized action. Use 'SURVEY <id>' to reveal a "
                "project's true value or 'SELECT <id,id,...>' to commit "
                "your final portfolio."
            )

        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            obs += (
                " Step limit reached; no portfolio was committed, episode "
                "ends with no additional reward."
            )
        else:
            obs += " Steps used: %d/%d." % (self.step_count, self.max_steps)

        self.terminated_flag = terminated or truncated
        return obs, reward, terminated, truncated, {}
