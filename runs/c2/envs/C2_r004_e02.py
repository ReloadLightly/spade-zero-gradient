import random


class MysteryDenominationKioskEnv:
    def __init__(self):
        self.rng = None

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.known_coins = [1, 5, 10, 25]
        self.candidates = sorted(self.rng.sample(range(6, 24), 5))
        self.mystery_value = self.rng.choice(self.candidates)
        self.target = self.rng.randint(28, 55)
        self.step_count = 0
        self.max_steps = 10
        self.probes_used = 0
        self.max_probes = 2
        self.consistent = list(self.candidates)
        self.milestones_hit = set()
        self.terminated = False
        obs = self._intro()
        return obs, {}

    def _intro(self):
        return (
            f"Kiosk change-making. Standard coins (unlimited supply): {self.known_coins}. "
            f"A mystery coin of unknown value M is also available; M is exactly one of "
            f"these candidates: {self.candidates}. Give exact change totaling {self.target} "
            f"using the FEWEST coins possible (any denominations, any quantities).\n"
            f"Actions (exactly one per turn):\n"
            f"  PROBE <v>   -- ask whether M >= v (yes/no answer); at most {self.max_probes} total\n"
            f"  COMMIT <c1>,<c5>,<c10>,<c25>,<cM>  -- final counts of coins 1,5,10,25,M; ends episode\n"
            f"You have {self.max_steps} steps total."
        )

    def _truncate_flag(self):
        if self.step_count >= self.max_steps:
            self.terminated = True
            return True
        return False

    def _optimal_coin_count(self):
        denoms = self.known_coins + [self.mystery_value]
        INF = float("inf")
        dp = [0] + [INF] * self.target
        for amt in range(1, self.target + 1):
            best = INF
            for d in denoms:
                if d <= amt and dp[amt - d] + 1 < best:
                    best = dp[amt - d] + 1
            dp[amt] = best
        return dp[self.target]

    def step(self, action):
        if self.terminated:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        action = (action or "").strip()
        upper = action.upper()

        if upper.startswith("PROBE"):
            parts = action.split()
            if len(parts) != 2 or not parts[1].isdigit():
                return ("Malformed PROBE. Use: PROBE <non-negative integer>.",
                        0.0, False, self._truncate_flag(), {})
            if self.probes_used >= self.max_probes:
                return (f"No probes remaining ({self.max_probes} already used).",
                        0.0, False, self._truncate_flag(), {})
            v = int(parts[1])
            self.probes_used += 1
            yes = self.mystery_value >= v
            before = len(self.consistent)
            if yes:
                self.consistent = [c for c in self.consistent if c >= v]
            else:
                self.consistent = [c for c in self.consistent if c < v]
            after = len(self.consistent)
            reward = 0.0
            if self.probes_used == 1 and after < before and "probe1" not in self.milestones_hit:
                reward += 0.15
                self.milestones_hit.add("probe1")
            if self.probes_used == 2 and after <= 3 and "probe2" not in self.milestones_hit:
                reward += 0.15
                self.milestones_hit.add("probe2")
            obs = (f"{'YES' if yes else 'NO'} (M {'>=' if yes else '<'} {v}). "
                   f"Consistent candidates: {self.consistent}. "
                   f"Probes left: {self.max_probes - self.probes_used}.")
            return obs, reward, False, self._truncate_flag(), {}

        if upper.startswith("COMMIT"):
            rest = action[len("COMMIT"):].strip()
            parts = [p.strip() for p in rest.split(",")]
            if len(parts) != 5 or not all(p.isdigit() for p in parts):
                return ("Malformed COMMIT. Use: COMMIT <c1>,<c5>,<c10>,<c25>,<cM> "
                        "with 5 non-negative integers.", 0.0, False, self._truncate_flag(), {})
            c1, c5, c10, c25, cM = (int(p) for p in parts)
            total = c1 * 1 + c5 * 5 + c10 * 10 + c25 * 25 + cM * self.mystery_value
            coin_count = c1 + c5 + c10 + c25 + cM
            self.terminated = True
            if total != self.target:
                obs = (f"COMMIT totals {total}, not the required {self.target}. Episode over. "
                       f"(The mystery coin was worth {self.mystery_value}.)")
                return obs, 0.0, True, False, {}
            reward = 0.2
            optimal = self._optimal_coin_count()
            diff = coin_count - optimal
            if diff <= 0:
                eff = 0.5
            elif diff == 1:
                eff = 0.35
            elif diff == 2:
                eff = 0.2
            else:
                eff = 0.05
            reward += eff
            obs = (f"COMMIT accepted: {coin_count} coins totaling {self.target}. "
                   f"The mystery coin was worth {self.mystery_value}. Optimal was {optimal} coins. "
                   f"Efficiency bonus: {eff:.2f}.")
            return obs, reward, True, False, {}

        return ("Unrecognized action. Use PROBE <v> or COMMIT <c1>,<c5>,<c10>,<c25>,<cM>.",
                0.0, False, self._truncate_flag(), {})
