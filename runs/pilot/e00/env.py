import random


class PotionHouseEnv:
    INGREDIENTS = ['A', 'B', 'C', 'D', 'E', 'F']
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.partner = {}
        self.tested_pairs = set()
        self.steps = 0
        self.done = False
        self.milestone_2 = False
        self.milestone_4 = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        shuffled = self.INGREDIENTS[:]
        self.rng.shuffle(shuffled)
        self.partner = {}
        for i in range(0, 6, 2):
            a, b = shuffled[i], shuffled[i + 1]
            self.partner[a] = b
            self.partner[b] = a

        self.tested_pairs = set()
        self.steps = 0
        self.done = False
        self.milestone_2 = False
        self.milestone_4 = False

        obs = (
            "You are a potion master. There are 6 ingredients: A, B, C, D, E, F.\n"
            "They belong to 3 secret Houses, two ingredients per House (Houses are "
            "unknown to you). Two ingredients form a STABLE pair only if they belong "
            "to DIFFERENT Houses; two ingredients from the SAME House are UNSTABLE and "
            "ruin a potion.\n"
            "GOAL: brew a valid potion by naming three ingredients that are pairwise "
            "STABLE (one ingredient from each of the 3 Houses).\n"
            "ACTIONS (send exactly one per turn):\n"
            "  TEST <X> <Y>       - probe whether two ingredients are STABLE or UNSTABLE\n"
            "  BREW <X> <Y> <Z>   - attempt the final potion with three ingredients\n"
            "You have at most 10 turns total. A correct BREW ends the episode "
            "successfully; an incorrect BREW reveals which of its pairs were unstable, "
            "and the episode continues if turns remain."
        )
        return obs, {}

    def _fmt_pair(self, x, y):
        return f"{x}{y}" if x < y else f"{y}{x}"

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        reward = 0.0
        terminated = False

        parts = action.strip().upper().split() if action else []
        cmd = parts[0] if parts else ""

        if cmd == "TEST" and len(parts) == 3:
            x, y = parts[1], parts[2]
            if x in self.INGREDIENTS and y in self.INGREDIENTS and x != y:
                pair_key = self._fmt_pair(x, y)
                self.tested_pairs.add(pair_key)
                same_house = self.partner[x] == y
                status = "UNSTABLE (same House)" if same_house else "STABLE (different Houses)"
                obs = (f"TEST {x} {y}: {status}. Distinct pairs tested so far: "
                       f"{len(self.tested_pairs)}.")
                if len(self.tested_pairs) >= 2 and not self.milestone_2:
                    self.milestone_2 = True
                    reward += 0.3
                    obs += " [Exploration credit +0.3]"
                if len(self.tested_pairs) >= 4 and not self.milestone_4:
                    self.milestone_4 = True
                    reward += 0.3
                    obs += " [Exploration credit +0.3]"
            else:
                obs = "Malformed TEST: give two different letters from A-F, e.g. 'TEST A C'."
        elif cmd == "BREW" and len(parts) == 4:
            trio = parts[1:4]
            if len(set(trio)) == 3 and all(t in self.INGREDIENTS for t in trio):
                x, y, z = trio
                bad_pairs = []
                for p, q in [(x, y), (x, z), (y, z)]:
                    if self.partner[p] == q:
                        bad_pairs.append(self._fmt_pair(p, q))
                if not bad_pairs:
                    reward += 0.4
                    terminated = True
                    self.done = True
                    obs = f"BREW {x} {y} {z}: all three pairs STABLE. The potion succeeds!"
                else:
                    obs = (f"BREW {x} {y} {z}: FAILED. Unstable pair(s) among your trio: "
                           f"{', '.join(bad_pairs)}.")
            else:
                obs = "Malformed BREW: give three different letters from A-F, e.g. 'BREW A C E'."
        else:
            obs = ("Unrecognized action. Use 'TEST <X> <Y>' or 'BREW <X> <Y> <Z>' "
                   "with letters from A-F.")

        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += " Step limit reached."

        info = {"steps": self.steps, "tested_pairs": len(self.tested_pairs)}
        return obs, reward, terminated, truncated, info
