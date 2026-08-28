import re


class TaintedMintEnv:
    KNOWN = [1, 5, 11, 25]
    TAINT_CANDIDATES = [5, 11, 25]
    MAX_STEPS = 10
    MAX_AUDITS = 1

    def __init__(self):
        self.rng = None
        self.target = 0
        self.tainted = None
        self.candidates = set()
        self.audits_used = 0
        self.steps = 0
        self.done = False

    def _min_coins(self, denoms, amount):
        INF = float("inf")
        dp = [0] + [INF] * amount
        for a in range(1, amount + 1):
            best = INF
            for d in denoms:
                if d <= a and dp[a - d] + 1 < best:
                    best = dp[a - d] + 1
            dp[a] = best
        return dp[amount]

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.target = self.rng.randint(38, 82)
        self.tainted = self.rng.choice(self.TAINT_CANDIDATES)
        self.candidates = set(self.TAINT_CANDIDATES)
        self.audits_used = 0
        self.steps = 0
        self.done = False
        obs = (
            f"MINT AUDIT: A cashier's drawer holds coins of value 1, 5, 11, 25 "
            f"(unlimited supply each). One of {{5, 11, 25}} has been secretly "
            f"recalled as counterfeit (the 1-coin reserve is never affected) — "
            f"you don't know which. You must pay exactly {self.target} using the "
            f"fewest coins possible, but if your payment includes even one coin "
            f"of the recalled value, the cashier refuses the whole payment and "
            f"the deal ends immediately with no further credit.\n"
            f"Actions:\n"
            f"  AUDIT <value>   - spend your ONE allowed audit (value must be "
            f"5, 11, or 25) to learn CLEAN or TAINTED for that value.\n"
            f"  COMMIT <v1>x<n1>,<v2>x<n2>,...  - pay with n1 coins of value v1, "
            f"etc. This ends the episode.\n"
            f"You have {self.MAX_STEPS} total steps and at most {self.MAX_AUDITS} "
            f"audit. Malformed actions are corrected with no reward and no "
            f"progress lost."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()
        reward = 0.0
        terminated = False
        info = {}

        m_audit = re.match(r"^AUDIT\s+(\d+)$", action, re.IGNORECASE)
        m_commit = re.match(r"^COMMIT\s+(.+)$", action, re.IGNORECASE)

        if m_audit:
            value = int(m_audit.group(1))
            if value not in self.TAINT_CANDIDATES:
                obs = f"Invalid audit target. Choose one of {self.TAINT_CANDIDATES}."
            elif self.audits_used >= self.MAX_AUDITS:
                obs = "No audits remaining."
            else:
                self.audits_used += 1
                before = len(self.candidates)
                if value == self.tainted:
                    self.candidates = {value}
                    result = "TAINTED"
                else:
                    self.candidates.discard(value)
                    result = "CLEAN"
                after = len(self.candidates)
                reduction = before - after
                narrow_reward = 0.15 * (reduction / 2.0)
                reward = 0.1 + narrow_reward
                remaining = sorted(self.candidates)
                obs = (
                    f"Audit result for {value}: {result}. Remaining suspect "
                    f"denominations: {remaining}."
                )

        elif m_commit:
            spec = m_commit.group(1)
            tokens = [t.strip() for t in spec.split(",") if t.strip()]
            parsed = []
            valid_syntax = True
            for tok in tokens:
                mt = re.match(r"^(\d+)x(\d+)$", tok)
                if not mt:
                    valid_syntax = False
                    break
                val, cnt = int(mt.group(1)), int(mt.group(2))
                if val not in self.KNOWN or cnt <= 0:
                    valid_syntax = False
                    break
                parsed.append((val, cnt))

            if not valid_syntax or not parsed:
                obs = (
                    "Malformed COMMIT. Use format like 25x2,5x1,1x3 with only "
                    "denominations 1, 5, 11, 25."
                )
            else:
                total = sum(v * c for v, c in parsed)
                if total != self.target:
                    obs = (
                        f"That sums to {total}, not the required {self.target}. "
                        f"Try again."
                    )
                elif any(v == self.tainted and c > 0 for v, c in parsed):
                    obs = (
                        f"REFUSED: your payment included recalled {self.tainted}-"
                        f"coins. The cashier voids the transaction."
                    )
                    terminated = True
                    reward = 0.0
                else:
                    used_coins = sum(c for _, c in parsed)
                    clean_denoms = [d for d in self.KNOWN if d != self.tainted]
                    true_optimal = self._min_coins(clean_denoms, self.target)
                    if used_coins == true_optimal:
                        band = 0.5
                    elif used_coins <= true_optimal + 1:
                        band = 0.3
                    elif used_coins <= true_optimal + 2:
                        band = 0.15
                    else:
                        band = 0.05
                    reward = 0.25 + band
                    terminated = True
                    obs = (
                        f"ACCEPTED: paid {self.target} using {used_coins} coins "
                        f"(true minimum with clean coins was {true_optimal}). Deal "
                        f"complete."
                    )
        else:
            obs = (
                "Unrecognized action. Use 'AUDIT <value>' or "
                "'COMMIT <v>x<n>,...'."
            )

        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            obs += " Step limit reached without a completed payment."

        self.done = terminated or truncated
        return obs, reward, terminated, truncated, info
