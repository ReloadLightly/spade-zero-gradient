import random


class GemMedianEnv:
    GEM_NAMES = ["A", "B", "C", "D", "E", "F", "G"]

    def __init__(self):
        self.rng = None
        self.weights = {}
        self.step_count = 0
        self.max_steps = 10
        self.done = False
        self.edges = {}
        self.milestone_awarded = False
        self.median_gem = None

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.done = False
        self.milestone_awarded = False
        values = list(range(1, 8))
        self.rng.shuffle(values)
        self.weights = dict(zip(self.GEM_NAMES, values))
        self.median_gem = next(g for g, w in self.weights.items() if w == 4)
        self.edges = {g: set() for g in self.GEM_NAMES}
        obs = (
            "You are a gem appraiser. Among gems A, B, C, D, E, F, G, exactly one is the "
            "authentic gem: the one whose true weight is exactly in the middle when all "
            "seven are ranked (heavier than exactly three others, lighter than exactly "
            "three others). All seven weights are distinct; no weighing ever balances.\n"
            "Actions:\n"
            "  WEIGH <gem1> <gem2> - place gem1 on the left pan, gem2 on the right pan; "
            "you learn which side is heavier.\n"
            "  SUBMIT <gem> - declare your final answer (ends the episode).\n"
            "You have 10 steps total."
        )
        return obs, {}

    def _direct_heavier(self, g1, g2):
        return self.weights[g1] > self.weights[g2]

    def _reachable(self, start):
        seen = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt in self.edges[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def _check_milestone(self):
        for g in self.GEM_NAMES:
            lighter = self._reachable(g)
            heavier = set()
            for other in self.GEM_NAMES:
                if other != g and g in self._reachable(other):
                    heavier.add(other)
            if len(lighter) == 3 and len(heavier) == 3:
                return g
        return None

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        parts = action.strip().split()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if (
            len(parts) == 3
            and parts[0].upper() == "WEIGH"
            and parts[1].upper() in self.weights
            and parts[2].upper() in self.weights
            and parts[1].upper() != parts[2].upper()
        ):
            g1, g2 = parts[1].upper(), parts[2].upper()
            if self._direct_heavier(g1, g2):
                self.edges[g1].add(g2)
                obs = f"LEFT_HEAVIER: {g1} is heavier than {g2}."
            else:
                self.edges[g2].add(g1)
                obs = f"RIGHT_HEAVIER: {g2} is heavier than {g1}."
            proven = self._check_milestone()
            if proven is not None and not self.milestone_awarded:
                self.milestone_awarded = True
                reward += 0.3
                obs += " (You now have enough evidence to prove which gem is authentic.)"
        elif len(parts) == 2 and parts[0].upper() == "SUBMIT" and parts[1].upper() in self.weights:
            guess = parts[1].upper()
            terminated = True
            self.done = True
            if guess == self.median_gem:
                reward += 0.7
                obs = f"Correct. {guess} is the authentic gem."
            else:
                obs = f"Incorrect. {guess} is not the authentic gem."
        else:
            obs = "Malformed action. Use 'WEIGH <gem> <gem>' or 'SUBMIT <gem>'."

        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            self.done = True
            obs += " Step limit reached."

        return obs, reward, terminated, truncated, info
