import random
import itertools


class FactoryLineSequencingEnv:
    def __init__(self):
        self.rng = None
        self.processing = []
        self.weights = []
        self.step_count = 0
        self.max_steps = 10
        self.asked_pairs = set()
        self.compare_rewards_given = 0
        self.done = False
        self._best_cost = 0
        self._worst_cost = 0

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.processing = [self.rng.randint(2, 9) for _ in range(5)]
        weight_pool = [1, 2, 3, 4, 5]
        self.rng.shuffle(weight_pool)
        self.weights = weight_pool
        self.step_count = 0
        self.asked_pairs = set()
        self.compare_rewards_given = 0
        self.done = False
        self._best_cost, self._worst_cost = self._bounds()

        lines = [
            "You control a single-line factory scheduler. 5 jobs (labeled 1-5) must each run",
            "exactly once, back-to-back with no idle time, in an order you choose.",
            "Known processing times (minutes): "
            + ", ".join(f"Job{i + 1}={p}" for i, p in enumerate(self.processing)) + ".",
            "Each job also carries a HIDDEN priority weight (a secret permutation of",
            "1,2,3,4,5, one value per job) representing its per-minute delay cost.",
            "GOAL: choose a full processing order minimizing total weighted completion time =",
            "sum over jobs of (weight_i * completion_time_i), where completion_time_i is the",
            "cumulative processing minutes up to and including job i's finish, in your order.",
            "ACTIONS (up to 10 total):",
            "  'COMPARE i j'  (i,j distinct, 1-5) -> reveals whether job i's hidden weight is",
            "      HIGHER or LOWER than job j's. The first 3 distinct pairs you query each earn",
            "      +0.1 (max +0.3 total); later queries cost a step but earn no reward.",
            "  'SUBMIT o1,o2,o3,o4,o5' -> commits a full order (permutation of 1-5), ends the",
            "      episode, and earns up to +0.7 based on how close your total weighted",
            "      completion time is to the true optimum (0.0 if as bad as the worst order).",
        ]
        return "\n".join(lines), {}

    def _cost(self, perm):
        total = 0
        elapsed = 0
        for idx in perm:
            elapsed += self.processing[idx]
            total += self.weights[idx] * elapsed
        return total

    def _bounds(self):
        best = worst = None
        for perm in itertools.permutations(range(5)):
            cost = self._cost(perm)
            if best is None or cost < best:
                best = cost
            if worst is None or cost > worst:
                worst = cost
        return best, worst

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        limit_hit = self.step_count >= self.max_steps
        text = (action or "").strip()
        head, _, rest = text.partition(" ")
        cmd = head.upper()

        if cmd == "COMPARE":
            args = rest.split()
            i = j = -1
            if len(args) == 2:
                try:
                    i = int(args[0]) - 1
                    j = int(args[1]) - 1
                except ValueError:
                    i = j = -1
            if 0 <= i < 5 and 0 <= j < 5 and i != j:
                reward = 0.0
                pair = frozenset((i, j))
                if pair not in self.asked_pairs and self.compare_rewards_given < 3:
                    self.compare_rewards_given += 1
                    reward = 0.1
                self.asked_pairs.add(pair)
                rel = "HIGHER" if self.weights[i] > self.weights[j] else "LOWER"
                obs = f"Job{i + 1}'s hidden weight is {rel} than Job{j + 1}'s."
                if limit_hit:
                    self.done = True
                    obs += " Step limit reached without a SUBMIT; episode ends with no order committed."
                return obs, reward, False, limit_hit, {}
            if limit_hit:
                self.done = True
            return (
                "Malformed COMPARE: use 'COMPARE i j' with two distinct job numbers from 1-5.",
                0.0, False, limit_hit, {},
            )

        if cmd == "SUBMIT":
            order_str = rest.replace(" ", "")
            order = []
            if order_str:
                try:
                    order = [int(x) - 1 for x in order_str.split(",")]
                except ValueError:
                    order = []
            if sorted(order) == list(range(5)):
                cost = self._cost(order)
                span = self._worst_cost - self._best_cost
                normalized = 1.0 if span <= 0 else (self._worst_cost - cost) / span
                normalized = max(0.0, min(1.0, normalized))
                reward = 0.7 * normalized
                self.done = True
                obs = (
                    f"Committed order {[o + 1 for o in order]}. Weighted completion time = "
                    f"{cost} (optimum = {self._best_cost}, worst = {self._worst_cost}). "
                    f"Score reward: {reward:.3f}."
                )
                return obs, reward, True, False, {}
            if limit_hit:
                self.done = True
            return (
                "Malformed SUBMIT: provide a comma-separated permutation of all 5 job "
                "numbers, e.g. 'SUBMIT 3,1,4,2,5'.",
                0.0, False, limit_hit, {},
            )

        if limit_hit:
            self.done = True
        return (
            "Unrecognized action. Use 'COMPARE i j' or 'SUBMIT o1,o2,o3,o4,o5'.",
            0.0, False, limit_hit, {},
        )
