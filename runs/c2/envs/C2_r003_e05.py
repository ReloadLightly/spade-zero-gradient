import math
import random


class FarmPlotAllocationEnv:
    ACRE_TOTAL = 10
    BLIGHT_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]
    WHEAT_COEFF = 6.0
    SOY_COEFF = 5.5
    CORN_COEFF = 5.0
    MAX_TESTS = 2
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.blight = None
        self.wheat_table = None
        self.soy_table = None
        self.corn_base_table = None
        self.corn_true_table = None
        self.steps = 0
        self.tests_used = 0
        self.done = False

    @staticmethod
    def _round_half_up(x):
        return int(math.floor(x + 0.5))

    def _build_table(self, coeff):
        return [self._round_half_up(coeff * math.sqrt(a)) for a in range(self.ACRE_TOTAL + 1)]

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.blight = self.rng.choice(self.BLIGHT_LEVELS)
        self.wheat_table = self._build_table(self.WHEAT_COEFF)
        self.soy_table = self._build_table(self.SOY_COEFF)
        self.corn_base_table = self._build_table(self.CORN_COEFF)
        self.corn_true_table = [self._round_half_up(v * self.blight) for v in self.corn_base_table]
        self.steps = 0
        self.tests_used = 0
        self.done = False
        return self._render_intro(), {}

    def _render_intro(self):
        lines = []
        lines.append(
            "You manage a %d-acre farm. Allocate ALL acres among WHEAT, CORN, and SOY "
            "(whole acres) to maximize total yield." % self.ACRE_TOTAL
        )
        lines.append("Known yield tables (index = acres, value = yield):")
        lines.append("WHEAT: " + str(self.wheat_table))
        lines.append("SOY:   " + str(self.soy_table))
        lines.append(
            "CORN is hit by an unknown blight this season. Its PRE-BLIGHT potential yield "
            "table is: " + str(self.corn_base_table)
        )
        lines.append(
            "Actual corn yield at acreage a = round(pre-blight[a] * severity), where severity "
            "is a hidden constant in {0.5, 0.6, 0.7, 0.8, 0.9} fixed for the whole season."
        )
        lines.append(
            "Actions (send exactly one per turn):\n"
            "  TEST 1          - plant a 1-acre corn test plot, see its real yield\n"
            "  TEST 2          - plant a 2-acre corn test plot, see its real yield\n"
            "  ALLOCATE w c s  - final whole-farm split (w+c+s must equal %d); ends the episode\n"
            "You may TEST at most %d times. You have %d steps total. Malformed actions are "
            "rejected, cost a step, and earn no reward." % (self.ACRE_TOTAL, self.MAX_TESTS, self.MAX_STEPS)
        )
        return "\n".join(lines)

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()
        parts = action.split()
        reward = 0.0
        terminated = False
        truncated = False

        if not parts:
            obs = "Empty action. Use 'TEST 1', 'TEST 2', or 'ALLOCATE w c s'."
        elif parts[0].upper() == "TEST" and len(parts) == 2 and parts[1] in ("1", "2"):
            if self.tests_used >= self.MAX_TESTS:
                obs = "No test plots remaining (used %d/%d)." % (self.tests_used, self.MAX_TESTS)
            else:
                a = int(parts[1])
                observed = self.corn_true_table[a]
                self.tests_used += 1
                reward = 0.1
                obs = "Test plot of %d corn acre(s) yielded %d. (%d/%d tests used)" % (
                    a, observed, self.tests_used, self.MAX_TESTS
                )
        elif parts[0].upper() == "ALLOCATE" and len(parts) == 4:
            try:
                w, c, s = int(parts[1]), int(parts[2]), int(parts[3])
                valid_range = all(0 <= v <= self.ACRE_TOTAL for v in (w, c, s))
            except ValueError:
                w = c = s = None
                valid_range = False

            if not valid_range or w + c + s != self.ACRE_TOTAL:
                obs = "Invalid allocation: acres must be non-negative integers summing to exactly %d." % self.ACRE_TOTAL
            else:
                reward += 0.1
                achieved = self.wheat_table[w] + self.corn_true_table[c] + self.soy_table[s]
                optimal = self._optimal_yield()
                ratio = achieved / optimal if optimal > 0 else 1.0
                if ratio >= 0.95:
                    tier = 0.7
                elif ratio >= 0.85:
                    tier = 0.45
                elif ratio >= 0.70:
                    tier = 0.2
                else:
                    tier = 0.0
                reward += tier
                terminated = True
                self.done = True
                obs = (
                    "ALLOCATE %d %d %d -> yield %d (wheat %d, corn %d, soy %d). True blight "
                    "severity was %.1f. Best possible yield was %d. Episode over."
                    % (w, c, s, achieved, self.wheat_table[w], self.corn_true_table[c],
                       self.soy_table[s], self.blight, optimal)
                )
        else:
            obs = (
                "Unrecognized action. Use 'TEST 1', 'TEST 2', or 'ALLOCATE w c s' "
                "(three integers summing to %d)." % self.ACRE_TOTAL
            )

        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += " Step limit reached; episode truncated."

        return obs, reward, terminated, truncated, {}

    def _optimal_yield(self):
        best = -1
        for w in range(self.ACRE_TOTAL + 1):
            for c in range(self.ACRE_TOTAL + 1 - w):
                s = self.ACRE_TOTAL - w - c
                total = self.wheat_table[w] + self.corn_true_table[c] + self.soy_table[s]
                if total > best:
                    best = total
        return best
