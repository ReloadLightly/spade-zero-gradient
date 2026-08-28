import random


class ChangeVaultEnv:
    MAX_STEPS = 10
    DENOM_LOW, DENOM_HIGH = 2, 30
    TARGET_LOW, TARGET_HIGH = 20, 50
    K = 4  # total distinct denominations; 1 is always one of them
    GEN_TRIES = 500

    def __init__(self):
        self.rng = None
        self.denoms = []
        self.target = 0
        self.dp_min = []
        self.step_count = 0
        self.done = False
        self.known_hits = set()
        self.discovery_awarded = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.done = False
        self.known_hits = set()
        self.discovery_awarded = False

        denoms, target, dp = None, None, None
        for _ in range(self.GEN_TRIES):
            others = set()
            while len(others) < self.K - 1:
                others.add(self.rng.randint(self.DENOM_LOW, self.DENOM_HIGH))
            cand_denoms = sorted(others | {1})
            if len(cand_denoms) != self.K:
                continue
            cand_target = self.rng.randint(self.TARGET_LOW, self.TARGET_HIGH)
            cand_dp = self._min_coins(cand_denoms, cand_target)
            greedy = self._greedy_coins(cand_denoms, cand_target)
            denoms, target, dp = cand_denoms, cand_target, cand_dp
            if greedy != cand_dp[cand_target]:
                break

        self.denoms, self.target, self.dp_min = denoms, target, dp

        obs = (
            f"COIN VAULT: this purse holds exactly {self.K} distinct coin denominations, "
            f"and 1 is always one of them; the other {self.K - 1} are hidden positive integers "
            f"(each at most {self.DENOM_HIGH}), with unlimited coins of every real denomination.\n"
            f"GOAL: make exact change for {self.target} using as few coins as possible.\n"
            f"ACTIONS: 'PROBE n' asks whether n is a real denomination (yes/no, costs a step). "
            f"'CHANGE c1,c2,...' submits your final coin list and ENDS the episode (costs a step).\n"
            f"You have {self.MAX_STEPS} steps total. Greedy (always take the largest coin that "
            f"fits) is NOT guaranteed optimal in this purse."
        )
        return obs, {}

    def _min_coins(self, denoms, target):
        dp = [0] + [None] * target
        for amt in range(1, target + 1):
            best = None
            for d in denoms:
                if d <= amt and dp[amt - d] is not None:
                    cand = dp[amt - d] + 1
                    if best is None or cand < best:
                        best = cand
            dp[amt] = best
        return dp

    def _greedy_coins(self, denoms, target):
        remaining, count = target, 0
        for d in sorted(denoms, reverse=True):
            if d <= remaining:
                take = remaining // d
                count += take
                remaining -= take * d
        return count if remaining == 0 else None

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        action = (action or "").strip()
        reward = 0.0
        terminated = False
        parts = action.split(None, 1)
        verb = parts[0].upper() if parts else ""

        if verb == "PROBE" and len(parts) == 2:
            try:
                n = int(parts[1].strip())
            except ValueError:
                obs = "Malformed PROBE: give one integer, e.g. a whole number after PROBE."
            else:
                if n == 1:
                    obs = "1 IS a denomination (you already knew that)."
                elif n in self.denoms:
                    obs = f"{n} IS a denomination."
                    self.known_hits.add(n)
                    if not self.discovery_awarded and len(self.known_hits) >= 2:
                        reward += 0.3
                        self.discovery_awarded = True
                else:
                    obs = f"{n} is NOT a denomination."
        elif verb == "CHANGE" and len(parts) == 2:
            terminated = True
            self.done = True
            try:
                coins = [int(x.strip()) for x in parts[1].split(",") if x.strip() != ""]
            except ValueError:
                coins = None
            if not coins:
                obs = "Malformed or empty CHANGE list; treated as a failed submission."
            else:
                total = sum(coins)
                all_real = all(c in self.denoms for c in coins)
                if total != self.target:
                    obs = f"Coins sum to {total}, not the target {self.target}. Failed."
                elif not all_real:
                    obs = "Submission includes a value that is not a real denomination. Failed."
                else:
                    optimal = self.dp_min[self.target]
                    used = len(coins)
                    if used == optimal:
                        reward += 0.7
                        obs = f"Correct and optimal: {used} coins (optimal is {optimal})."
                    elif used == optimal + 1:
                        reward += 0.35
                        obs = f"Valid, one coin above optimal: {used} coins (optimal {optimal})."
                    else:
                        reward += 0.1
                        obs = f"Valid but far from optimal: {used} coins (optimal {optimal})."
        else:
            obs = "Malformed action. Use 'PROBE n' or 'CHANGE c1,c2,...'."

        truncated = False
        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += f" Step budget ({self.MAX_STEPS}) exhausted."
        elif not terminated:
            obs += f" [step {self.step_count}/{self.MAX_STEPS}]"

        return obs, reward, terminated, truncated, {}
