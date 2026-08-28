import random


class NightBridgePlankEnv:
    """One rotten plank hidden among 8; two torch checks (group tests) give
    CREAK/SOLID verdicts; then a single irreversible CROSS commit."""

    N_PLANKS = 8
    MAX_CHECKS = 2
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.rotten = self.rng.randint(1, self.N_PLANKS)
        self.history = []  # list of (frozenset(subset), "CREAK"/"SOLID")
        self.checks_used = 0
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.m1_awarded = False
        self.m2_awarded = False

        obs = (
            "NIGHT BRIDGE PLANK CHECK\n"
            f"A rope bridge has {self.N_PLANKS} numbered planks (1-{self.N_PLANKS}). "
            "Exactly one is secretly rotten and will give way if you cross on it. "
            "Your torch has fuel for exactly 2 checks before it gutters out.\n"
            "Actions (one per turn, up to 10 turns total):\n"
            "  CHECK a,b,c  -- shine the torch over a group of plank numbers; "
            "returns CREAK (the rotten plank is somewhere in that group) or "
            "SOLID (it is not).\n"
            "  CROSS n      -- commit to crossing, naming the one plank you believe "
            "is rotten so you can step around it. This ends the episode.\n"
            "A CHECK group must be a proper subset (not empty, not all planks). "
            "Malformed actions are corrected with no reward and no torch fuel lost, "
            "but still use a turn."
        )
        info = {"step": self.step_count, "torch_fuel_left": self.MAX_CHECKS}
        return obs, info

    def _candidates(self, history):
        cands = []
        for p in range(1, self.N_PLANKS + 1):
            ok = True
            for subset, result in history:
                in_group = p in subset
                said_creak = (result == "CREAK")
                if in_group != said_creak:
                    ok = False
                    break
            if ok:
                cands.append(p)
        return cands

    def _parse_int_list(self, text):
        parts = [t.strip() for t in text.split(",") if t.strip() != ""]
        nums = []
        for t in parts:
            if not t.lstrip("-").isdigit():
                return None
            n = int(t)
            if n < 1 or n > self.N_PLANKS:
                return None
            nums.append(n)
        if len(nums) == 0 or len(set(nums)) != len(nums):
            return None
        return nums

    def step(self, action):
        if self.terminated or self.truncated:
            return "Episode already over.", 0.0, self.terminated, self.truncated, {}

        self.step_count += 1
        text = (action or "").strip()
        upper = text.upper()
        reward = 0.0
        obs = ""

        if upper.startswith("CHECK"):
            rest = text[5:].strip()
            nums = self._parse_int_list(rest)
            if nums is None or len(nums) >= self.N_PLANKS:
                obs = (
                    "Malformed CHECK. Use: CHECK a,b,c with 1 to "
                    f"{self.N_PLANKS - 1} distinct plank numbers between 1 and "
                    f"{self.N_PLANKS}."
                )
            elif self.checks_used >= self.MAX_CHECKS:
                obs = "The torch gutters and dies -- no checks remain. You must CROSS."
            else:
                subset = frozenset(nums)
                candidates_before = self._candidates(self.history)
                result = "CREAK" if self.rotten in subset else "SOLID"
                self.history.append((subset, result))
                self.checks_used += 1
                candidates_after = self._candidates(self.history)

                obs = (
                    f"You check planks {sorted(subset)} by torchlight: {result}. "
                    f"Torch checks left: {self.MAX_CHECKS - self.checks_used}."
                )

                if (
                    self.checks_used == 1
                    and not self.m1_awarded
                    and len(candidates_after) < len(candidates_before)
                    and len(candidates_after) <= 4
                ):
                    reward = 0.15
                    self.m1_awarded = True
                elif (
                    self.checks_used == 2
                    and not self.m2_awarded
                    and len(candidates_after) < len(candidates_before)
                    and len(candidates_after) <= 2
                ):
                    reward = 0.15
                    self.m2_awarded = True

        elif upper.startswith("CROSS"):
            rest = text[5:].strip()
            nums = self._parse_int_list(rest)
            if nums is None or len(nums) != 1:
                obs = f"Malformed CROSS. Use: CROSS n with a single plank number 1-{self.N_PLANKS}."
            else:
                guess = nums[0]
                candidates = self._candidates(self.history)
                self.terminated = True
                if guess == self.rotten:
                    reward = 0.7
                    obs = (
                        f"You step around plank {guess} -- it CREAKS and gives way as you pass! "
                        "You judged it correctly and cross safely. Episode over."
                    )
                elif guess in candidates:
                    reward = 0.3
                    obs = (
                        f"You step around plank {guess}, but the real weak plank is elsewhere. "
                        "It groans under your foot -- you scramble across, shaken but alive. "
                        "Episode over."
                    )
                else:
                    reward = 0.0
                    obs = (
                        f"You confidently avoid plank {guess}, but your checks had already "
                        "proven that plank sound -- the real rotten one catches you off guard. "
                        "Episode over."
                    )

        else:
            obs = "Unrecognized action. Use CHECK a,b,c or CROSS n."

        if not self.terminated and self.step_count >= self.MAX_STEPS:
            self.truncated = True
            obs += " The last of the night slips away -- you never committed to a crossing."

        info = {
            "step": self.step_count,
            "torch_fuel_left": max(0, self.MAX_CHECKS - self.checks_used),
        }
        return obs, reward, self.terminated, self.truncated, info
