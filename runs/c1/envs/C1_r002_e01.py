import math
import random


class ConjunctionAlmanacEnv:
    """Infer two hidden orbital periods from coarse telescope readings and
    predict all conjunction days within a fixed observing window."""

    HORIZON = 40
    PERIOD_MIN = 4
    PERIOD_MAX = 18
    MAX_LCM = 40
    NUM_SECTORS = 8
    MAX_STEPS = 9  # probes + one final SUBMIT

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        candidates = list(range(self.PERIOD_MIN, self.PERIOD_MAX + 1))
        while True:
            pa, pb = self.rng.sample(candidates, 2)
            lcm = pa * pb // math.gcd(pa, pb)
            if lcm <= self.MAX_LCM:
                break
        self.pa = pa
        self.pb = pb
        self.lcm = lcm
        self.true_conjunctions = set(range(self.lcm, self.HORIZON + 1, self.lcm))
        self.step_count = 0
        self.done = False

        obs = (
            "TWIN-PLANET OBSERVATORY\n"
            "Two planets, Planet-1 and Planet-2, orbit a star with fixed but "
            "unknown integer periods (each between 4 and 18 days). At day 0 "
            "both planets sit exactly at sector 0 (a conjunction). Your "
            "telescope reports each planet's coarse position as one of 8 "
            "sectors (0-7), sweeping around the orbit once per period.\n\n"
            "GOAL: determine each planet's exact orbital period, then list "
            "every day from 1 to 40 (inclusive) on which the two planets are "
            "again in conjunction (i.e. day is a multiple of BOTH periods).\n\n"
            "ACTIONS (exactly one per turn):\n"
            "  PROBE <day>            -- point the telescope at day 1-40, "
            "get both planets' sectors\n"
            "  SUBMIT <P1> <P2> <d1> <d2> ...  -- final answer: your guess "
            "for Planet-1's period, Planet-2's period, then every "
            "conjunction day you believe occurs in [1,40] (space-separated; "
            "may be empty)\n\n"
            f"You have {self.MAX_STEPS} actions total (probes + the final "
            "SUBMIT), so spend them deliberately -- scanning every day "
            "one-by-one will run you out of budget for periods this large.\n"
            "SUBMIT ends the episode."
        )
        return obs, {}

    def _sector(self, day, period):
        r = day % period
        return (r * self.NUM_SECTORS) // period

    def step(self, action):
        self.step_count += 1
        remaining = self.MAX_STEPS - self.step_count
        text = (action or "").strip()
        parts = text.split()

        if not parts:
            obs = "Empty action. Use 'PROBE <day>' or 'SUBMIT <P1> <P2> <d1> ...'."
            return obs, 0.0, False, remaining <= 0, {}

        cmd = parts[0].upper()

        if cmd == "PROBE":
            if len(parts) != 2:
                obs = "Malformed PROBE. Use exactly: PROBE <day>"
                return obs, 0.0, False, remaining <= 0, {}
            try:
                day = int(parts[1])
            except ValueError:
                obs = "Malformed PROBE. <day> must be an integer."
                return obs, 0.0, False, remaining <= 0, {}
            if not (1 <= day <= self.HORIZON):
                obs = f"Day out of range. Choose a day between 1 and {self.HORIZON}."
                return obs, 0.0, False, remaining <= 0, {}
            s1 = self._sector(day, self.pa)
            s2 = self._sector(day, self.pb)
            obs = (
                f"Day {day}: Planet-1 sector {s1}, Planet-2 sector {s2}. "
                f"({remaining} actions left after this one.)"
            )
            return obs, 0.0, False, remaining <= 0, {}

        if cmd == "SUBMIT":
            if len(parts) < 3:
                obs = "Malformed SUBMIT. Use: SUBMIT <P1> <P2> <d1> <d2> ..."
                return obs, 0.0, False, remaining <= 0, {}
            try:
                nums = [int(p) for p in parts[1:]]
            except ValueError:
                obs = "Malformed SUBMIT. All arguments must be integers."
                return obs, 0.0, False, remaining <= 0, {}

            pa_guess, pb_guess = nums[0], nums[1]
            submitted_days = set(nums[2:])

            reward = 0.0
            if pa_guess == self.pa:
                reward += 0.25
            if pb_guess == self.pb:
                reward += 0.25

            union = submitted_days | self.true_conjunctions
            inter = submitted_days & self.true_conjunctions
            jaccard = (len(inter) / len(union)) if union else 1.0
            reward += 0.5 * jaccard
            reward = min(reward, 1.0)

            obs = (
                f"Submission recorded. Planet-1 period {'correct' if pa_guess == self.pa else 'incorrect'}, "
                f"Planet-2 period {'correct' if pb_guess == self.pb else 'incorrect'}. "
                f"Conjunction days matched {len(inter)} of {len(self.true_conjunctions)} true days "
                f"(you listed {len(submitted_days)}). Final reward: {reward:.2f}."
            )
            self.done = True
            return obs, reward, True, False, {
                "true_pa": self.pa,
                "true_pb": self.pb,
                "true_conjunctions": sorted(self.true_conjunctions),
            }

        obs = "Unknown command. Use 'PROBE <day>' or 'SUBMIT <P1> <P2> <d1> ...'."
        return obs, 0.0, False, remaining <= 0, {}
