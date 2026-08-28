import random
import re


class CoinVaultEnv:
    """Make exact change from a limited-supply vault using the fewest coins."""

    POOL = [1, 2, 5, 10, 20, 25, 50]
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.denoms = sorted(self.rng.sample(self.POOL, 4), reverse=True)
        self.supply = {v: self.rng.randint(1, 4) for v in self.denoms}
        while True:
            counts = {v: self.rng.randint(0, self.supply[v]) for v in self.denoms}
            total = sum(v * c for v, c in counts.items())
            if total > 0:
                break
        self.target = total
        self.optimal_count = self._compute_optimal()
        self.steps = 0
        self.milestone_a = False
        self.done = False
        return self._intro(), {}

    def _compute_optimal(self):
        t = self.target
        inf = float("inf")
        dp = [0] + [inf] * t
        for v in self.denoms:
            s = self.supply[v]
            ndp = dp[:]
            for amt in range(t + 1):
                if dp[amt] == inf:
                    continue
                for k in range(1, s + 1):
                    na = amt + k * v
                    if na > t:
                        break
                    if dp[amt] + k < ndp[na]:
                        ndp[na] = dp[amt] + k
            dp = ndp
        return dp[t]

    def _intro(self):
        denom_str = ", ".join(str(v) for v in self.denoms)
        return (
            f"COIN VAULT. Available coin denominations: {denom_str}. "
            f"Target amount to pay exactly: {self.target}. "
            f"Each denomination has a hidden, limited supply in the vault; the same "
            f"denomination's supply never changes during the episode.\n"
            f"Goal: pay exactly {self.target} using a multiset of these coins, drawn "
            f"from the vault's real supply, using as FEW total coins as possible.\n"
            f"Actions (exactly one per turn):\n"
            f"  QUERY <value> - reveals how many coins of that denomination remain "
            f"in the vault. No reward.\n"
            f"  PROPOSE <value>:<count>,<value>:<count>,... - attempt to pay the "
            f"target with that multiset. Each denomination listed at most once.\n"
            f"You have {self.MAX_STEPS} steps total (QUERY and PROPOSE both count)."
        )

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        obs = None
        reward = 0.0
        terminated = False

        a = (action or "").strip()
        m_query = re.fullmatch(r"(?i)QUERY\s+(\d+)", a)
        m_propose = re.fullmatch(
            r"(?i)PROPOSE\s+(\d+:\d+(?:\s*,\s*\d+:\d+)*)", a
        )

        if m_query:
            value = int(m_query.group(1))
            if value not in self.denoms:
                obs = (
                    f"No such denomination in this vault: {value}. Valid "
                    f"denominations are {', '.join(str(v) for v in self.denoms)}."
                )
            else:
                obs = f"Denomination {value} has {self.supply[value]} coin(s) remaining in the vault."

        elif m_propose:
            pairs = [p.strip() for p in m_propose.group(1).split(",")]
            proposal = {}
            malformed = False
            for p in pairs:
                v_str, c_str = p.split(":")
                v, c = int(v_str), int(c_str)
                if v not in self.denoms or c <= 0 or v in proposal:
                    malformed = True
                    break
                proposal[v] = c

            if malformed:
                obs = (
                    f"Invalid PROPOSE: use only denominations {', '.join(str(v) for v in self.denoms)}, "
                    f"each listed once, with positive counts."
                )
            else:
                total = sum(v * c for v, c in proposal.items())
                diff = self.target - total
                if diff != 0:
                    direction = "under" if diff > 0 else "over"
                    obs = (
                        f"Sum is {total}, which is {abs(diff)} {direction} the target "
                        f"of {self.target}."
                    )
                else:
                    violations = {
                        v: (c, self.supply[v])
                        for v, c in proposal.items()
                        if c > self.supply[v]
                    }
                    if violations:
                        detail = "; ".join(
                            f"denomination {v} requested {req} but only {have} available"
                            for v, (req, have) in violations.items()
                        )
                        bonus = 0.0
                        if not self.milestone_a:
                            self.milestone_a = True
                            bonus = 0.2
                        reward = bonus
                        obs = (
                            f"Sum matches the target ({self.target}) but the vault "
                            f"can't supply it: {detail}."
                        )
                    else:
                        achieved = sum(proposal.values())
                        ratio = self.optimal_count / achieved
                        bonus = 0.0
                        if not self.milestone_a:
                            self.milestone_a = True
                            bonus = 0.2
                        reward = bonus + 0.8 * ratio
                        terminated = True
                        self.done = True
                        obs = (
                            f"Valid exact payment of {self.target} using {achieved} "
                            f"coin(s). Episode complete."
                        )
        else:
            obs = (
                "Malformed action. Use 'QUERY <value>' or "
                "'PROPOSE <value>:<count>,<value>:<count>,...'."
            )

        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += f" Step limit ({self.MAX_STEPS}) reached; episode ends."

        if not terminated and not truncated:
            obs += f" ({self.MAX_STEPS - self.steps} step(s) remaining.)"

        return obs, reward, terminated, truncated, {}
