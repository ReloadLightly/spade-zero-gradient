import random
import re


class BlockedAisleShelfEnv:
    AISLES = ['A', 'B', 'C', 'D']
    ITEMS = ['1', '2', '3', '4']
    PENALTY = 6
    MAX_TESTS = 2
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.nominal = {}
        self.freq = {}
        self.blocked = None
        self.tests_used = 0
        self.step_count = 0
        self.done = False
        self.m1_awarded = False
        self.m2_awarded = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        distances = self.rng.sample(range(3, 13), 4)
        self.nominal = dict(zip(self.AISLES, distances))
        freqs = self.rng.sample(range(1, 10), 4)
        self.freq = dict(zip(self.ITEMS, freqs))
        self.blocked = self.rng.choice(self.AISLES)
        self.tests_used = 0
        self.step_count = 0
        self.done = False
        self.m1_awarded = False
        self.m2_awarded = False

        dist_lines = ', '.join(f"{a}:{self.nominal[a]}m" for a in self.AISLES)
        freq_lines = ', '.join(f"item{i}:{self.freq[i]}/day" for i in self.ITEMS)
        obs = (
            "WAREHOUSE SHELF PLACEMENT\n"
            "Goal: assign 4 items to 4 aisle slots (A-D) to minimize total daily "
            "walking cost = sum(pick frequency x actual distance).\n"
            f"Nominal round-trip distances: {dist_lines}\n"
            f"One aisle is blocked by a stray cart today, adding +{self.PENALTY}m to "
            "that aisle's slot only. You do not know which aisle.\n"
            f"Item pick frequencies: {freq_lines}\n"
            "Actions (one per turn, 10 turns max):\n"
            "  TEST <letter>  - probe one aisle, up to 2 uses total. Returns SLOW if "
            "that aisle is the blocked one, else CLEAR.\n"
            "  COMMIT <i><s> <i><s> <i><s> <i><s>  - assign every item to a distinct "
            "slot, e.g. COMMIT 1A 2B 3C 4D. Ends the episode.\n"
            "Commit exactly once, when ready."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip().upper()
        tokens = text.split()
        reward = 0.0
        terminated = False

        if not tokens:
            obs = "Empty action. Use TEST <letter> or COMMIT <i><s> <i><s> <i><s> <i><s>."
        elif tokens[0] == "TEST" and len(tokens) == 2 and tokens[1] in self.AISLES:
            if self.tests_used >= self.MAX_TESTS:
                obs = f"No test budget remaining ({self.MAX_TESTS} used)."
            else:
                letter = tokens[1]
                self.tests_used += 1
                if not self.m1_awarded:
                    self.m1_awarded = True
                    reward += 0.15
                elif not self.m2_awarded:
                    self.m2_awarded = True
                    reward += 0.15
                result = "SLOW" if letter == self.blocked else "CLEAR"
                obs = f"TEST {letter}: {result}. Tests remaining: {self.MAX_TESTS - self.tests_used}."
        elif tokens[0] == "COMMIT":
            pairs = tokens[1:]
            valid = len(pairs) == 4 and all(re.fullmatch(r'[1-4][A-D]', p) for p in pairs)
            assign = {}
            if valid:
                for p in pairs:
                    assign[p[0]] = p[1]
                valid = (set(assign.keys()) == set(self.ITEMS)
                          and set(assign.values()) == set(self.AISLES))
            if not valid:
                obs = ("Invalid COMMIT: give exactly 4 pairs covering items 1-4 and "
                       "slots A-D exactly once each, e.g. COMMIT 1A 2B 3C 4D.")
            else:
                reward += 0.3
                actual = {a: self.nominal[a] + (self.PENALTY if a == self.blocked else 0)
                          for a in self.AISLES}
                total_cost = sum(self.freq[i] * actual[assign[i]] for i in self.ITEMS)

                items_by_freq = sorted(self.ITEMS, key=lambda i: -self.freq[i])
                slots_asc = sorted(self.AISLES, key=lambda a: actual[a])
                slots_desc = sorted(self.AISLES, key=lambda a: -actual[a])
                optimal_cost = sum(self.freq[items_by_freq[k]] * actual[slots_asc[k]]
                                    for k in range(4))
                worst_cost = sum(self.freq[items_by_freq[k]] * actual[slots_desc[k]]
                                  for k in range(4))

                if worst_cost - optimal_cost <= 0:
                    quality = 1.0 if total_cost <= optimal_cost else 0.0
                else:
                    quality = (worst_cost - total_cost) / (worst_cost - optimal_cost)
                    quality = max(0.0, min(1.0, quality))
                reward += 0.4 * quality

                terminated = True
                self.done = True
                obs = (
                    f"COMMIT accepted. Blocked aisle was {self.blocked} "
                    f"(+{self.PENALTY}m). Your cost: {total_cost}. "
                    f"Optimal: {optimal_cost}. Worst-case: {worst_cost}. "
                    f"Quality score: {quality:.2f}."
                )
        else:
            obs = ("Unrecognized action. Use TEST <letter> or "
                   "COMMIT <i><s> <i><s> <i><s> <i><s>.")

        truncated = False
        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += " Step limit reached without a commit; episode ends."

        return obs, reward, terminated, truncated, {
            "tests_used": self.tests_used,
            "step_count": self.step_count,
        }
