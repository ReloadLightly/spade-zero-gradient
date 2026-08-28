import random
import re


class MoneylendersVaultEnv:
    DENOM_SETS = [
        [1, 3, 4],
        [1, 4, 5, 6],
        [1, 4, 6, 9],
        [1, 3, 5, 6],
        [1, 4, 6, 7],
    ]
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.denoms = sorted(self.rng.choice(self.DENOM_SETS))
        self.target = self.rng.randint(16, 26)
        self.steps = 0
        self.finished = False
        self.sum_milestone_awarded = False

        unbounded_min, counts = self._unbounded_optimal(self.target, self.denoms)

        supply = {}
        non_one = [d for d in self.denoms if d != 1]
        constrained = max(non_one) if non_one else None
        for d in self.denoms:
            if d == 1:
                supply[d] = self.target
            elif d == constrained:
                supply[d] = max(0, counts.get(d, 0) - 1)
            else:
                supply[d] = self.rng.randint(2, 5)
        self.supply = supply
        self.true_min = self._bounded_optimal(self.target, self.denoms, supply)

        denom_list = ", ".join(str(d) for d in self.denoms)
        obs = (
            f"MONEYLENDER'S VAULT. Coin denominations in this system: {denom_list} "
            f"silver pieces. You must pay exactly {self.target} silver pieces total, "
            f"using as FEW coins as possible.\n"
            f"The vault holds a limited, hidden stock of each denomination — you only "
            f"learn a coin's remaining stock if you try to use more of it than is left.\n"
            f"Action format: 'PAY v1 v2 v3 ...' listing the coin values you spend, e.g. "
            f"a payment using two of one denomination and one of another.\n"
            f"You have {self.MAX_STEPS} steps total. Each attempt tells you whether your "
            f"total is too high or too low, and by how much; once your total is exactly "
            f"right, any denomination you over-used will be reported with its exact "
            f"remaining stock."
        )
        return obs, {}

    def step(self, action):
        if self.finished:
            return "Episode already ended.", 0.0, True, False, {}

        self.steps += 1
        reward = 0.0
        terminated = False

        parts = action.strip().split()
        if not parts or parts[0].upper() != "PAY":
            obs = "Malformed action. Use: PAY v1 v2 v3 ... (space-separated coin values)."
            truncated = self.steps >= self.MAX_STEPS
            self.finished = terminated or truncated
            return obs, 0.0, False, truncated, {}

        nums = re.findall(r"\d+", " ".join(parts[1:]))
        if not nums:
            obs = "No coin values found. Use: PAY v1 v2 v3 ..."
            truncated = self.steps >= self.MAX_STEPS
            self.finished = truncated
            return obs, 0.0, False, truncated, {}

        coins = [int(n) for n in nums]
        invalid = sorted(set(c for c in coins if c not in self.denoms))
        if invalid:
            obs = (
                f"Invalid coin(s): {invalid}. Valid denominations are: {self.denoms}."
            )
            truncated = self.steps >= self.MAX_STEPS
            self.finished = truncated
            return obs, 0.0, False, truncated, {}

        total = sum(coins)
        if total != self.target:
            diff = self.target - total
            if diff > 0:
                obs = f"Total is {total}, which is {diff} short of the target {self.target}."
            else:
                obs = f"Total is {total}, which is {-diff} over the target {self.target}."
            truncated = self.steps >= self.MAX_STEPS
            self.finished = truncated
            return obs, 0.0, False, truncated, {}

        if not self.sum_milestone_awarded:
            reward += 0.2
            self.sum_milestone_awarded = True

        counts = {}
        for c in coins:
            counts[c] = counts.get(c, 0) + 1

        violations = [
            (d, counts[d], self.supply[d]) for d in counts if counts[d] > self.supply[d]
        ]
        if violations:
            details = "; ".join(
                f"coin {d}: you used {used}, but only {left} remain in the vault"
                for d, used, left in violations
            )
            obs = (
                f"Total matches the target, but the vault can't cover it: {details}. "
                f"Try a different combination that respects the vault's stock."
            )
            truncated = self.steps >= self.MAX_STEPS
            self.finished = truncated
            return obs, reward, False, truncated, {}

        coins_used = len(coins)
        optimality_bonus = 0.6 * min(1.0, self.true_min / coins_used)
        reward += 0.2 + optimality_bonus
        terminated = True
        self.finished = True
        obs = (
            f"Accepted! You paid exactly {self.target} using {coins_used} coins "
            f"(the vault's true minimum given its hidden stock was {self.true_min})."
        )
        return obs, reward, terminated, False, {}

    @staticmethod
    def _unbounded_optimal(target, denoms):
        INF = float("inf")
        dp = [0] + [INF] * target
        parent = [-1] * (target + 1)
        for s in range(1, target + 1):
            for d in denoms:
                if d <= s and dp[s - d] + 1 < dp[s]:
                    dp[s] = dp[s - d] + 1
                    parent[s] = d
        counts = {d: 0 for d in denoms}
        s = target
        while s > 0 and parent[s] != -1:
            d = parent[s]
            counts[d] += 1
            s -= d
        return dp[target], counts

    @staticmethod
    def _bounded_optimal(target, denoms, supply):
        INF = float("inf")
        dp = [0] + [INF] * target
        for d in denoms:
            for _ in range(supply[d]):
                for s in range(target, d - 1, -1):
                    if dp[s - d] + 1 < dp[s]:
                        dp[s] = dp[s - d] + 1
        return dp[target]
