import random


class CargoBayManifestEnv:
    """Optimization game: pack hidden-weight boxes into the minimum number of bins."""

    N_ITEMS = 6
    CAPACITY = 10
    MAX_STEPS = 10
    WEIGH_BUDGET = 3
    MIN_WEIGHT = 2
    MAX_WEIGHT = 8

    def __init__(self):
        self.rng = None
        self.weights = []
        self.tags = []
        self.optimal = None
        self.steps = 0
        self.weighs_used = 0
        self.done = False

    def _tag(self, w):
        if w <= 3:
            return "light"
        if w <= 6:
            return "medium"
        return "heavy"

    def _min_bins(self, weights):
        n = len(weights)
        ws = sorted(weights, reverse=True)
        best = [n]

        def backtrack(idx, bins):
            if len(bins) >= best[0]:
                return
            if idx == n:
                best[0] = min(best[0], len(bins))
                return
            w = ws[idx]
            seen_caps = set()
            for bi in range(len(bins)):
                if bins[bi] >= w and bins[bi] not in seen_caps:
                    seen_caps.add(bins[bi])
                    bins[bi] -= w
                    backtrack(idx + 1, bins)
                    bins[bi] += w
            bins.append(self.CAPACITY - w)
            backtrack(idx + 1, bins)
            bins.pop()

        backtrack(0, [])
        return best[0]

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        while True:
            weights = [self.rng.randint(self.MIN_WEIGHT, self.MAX_WEIGHT) for _ in range(self.N_ITEMS)]
            optimal = self._min_bins(weights)
            if 2 <= optimal <= 4 and len(set(weights)) >= 3:
                break
        self.weights = weights
        self.tags = [self._tag(w) for w in weights]
        self.optimal = optimal
        self.steps = 0
        self.weighs_used = 0
        self.done = False

        tag_str = ", ".join(f"item{i}={self.tags[i]}" for i in range(self.N_ITEMS))
        obs = (
            f"CARGO BAY MANIFEST\n"
            f"Goal: pack all {self.N_ITEMS} boxes (item0..item{self.N_ITEMS - 1}) into the fewest "
            f"bins possible. Each bin holds total weight up to {self.CAPACITY}.\n"
            f"Coarse weight tags (light=2-3, medium=4-6, heavy=7-8): {tag_str}\n"
            f"Exact weights are hidden. Actions (one per step, {self.MAX_STEPS} steps total):\n"
            f"  WEIGH <item_id>       - reveal the exact weight of one item "
            f"(budget: {self.WEIGH_BUDGET} uses total)\n"
            f"  COMPARE <id> <id>     - learn '<', '>', or '=' between two items' exact weights\n"
            f"  PACK <b0>,<b1>,...    - submit one bin label per item, in item order "
            f"(e.g. PACK 0,0,1,1,2,0); episode ends on the first valid submission\n"
            f"An invalid PACK (wrong item count, non-bin token, or a bin over capacity) is "
            f"rejected with feedback and does not end the episode, but still costs a step."
        )
        info = {"steps_remaining": self.MAX_STEPS}
        return obs, info

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()
        parts = action.split()

        if not parts:
            return self._malformed("Empty action. Use WEIGH, COMPARE, or PACK.")

        verb = parts[0].upper()

        if verb == "WEIGH":
            return self._do_weigh(parts)
        if verb == "COMPARE":
            return self._do_compare(parts)
        if verb == "PACK":
            return self._do_pack(action)
        return self._malformed(f"Unknown action '{parts[0]}'. Use WEIGH, COMPARE, or PACK.")

    def _malformed(self, msg):
        obs = msg
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            obs += f" Step limit ({self.MAX_STEPS}) reached without a valid PACK."
        return obs, 0.0, False, truncated, {"steps_remaining": max(0, self.MAX_STEPS - self.steps)}

    def _do_weigh(self, parts):
        if len(parts) != 2 or not parts[1].isdigit():
            return self._malformed("WEIGH needs one item id, e.g. 'WEIGH 3'.")
        idx = int(parts[1])
        if not (0 <= idx < self.N_ITEMS):
            return self._malformed(f"item id must be 0..{self.N_ITEMS - 1}.")
        if self.weighs_used >= self.WEIGH_BUDGET:
            return self._malformed("WEIGH budget exhausted; use COMPARE instead.")
        self.weighs_used += 1
        obs = f"item{idx} exact weight = {self.weights[idx]}. (weighs used: {self.weighs_used}/{self.WEIGH_BUDGET})"
        return self._continue(obs)

    def _do_compare(self, parts):
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            return self._malformed("COMPARE needs two item ids, e.g. 'COMPARE 1 4'.")
        a, b = int(parts[1]), int(parts[2])
        if not (0 <= a < self.N_ITEMS and 0 <= b < self.N_ITEMS) or a == b:
            return self._malformed(f"item ids must be distinct and in 0..{self.N_ITEMS - 1}.")
        wa, wb = self.weights[a], self.weights[b]
        sym = "=" if wa == wb else ("<" if wa < wb else ">")
        obs = f"item{a} {sym} item{b}"
        return self._continue(obs)

    def _continue(self, obs):
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            obs += f" Step limit ({self.MAX_STEPS}) reached without a valid PACK."
        return obs, 0.0, False, truncated, {"steps_remaining": max(0, self.MAX_STEPS - self.steps)}

    def _do_pack(self, action):
        rest = action[len("PACK"):].strip()
        tokens = [t.strip() for t in rest.split(",") if t.strip() != ""]
        if len(tokens) != self.N_ITEMS:
            return self._malformed(
                f"PACK needs exactly {self.N_ITEMS} comma-separated bin labels, one per item."
            )
        bins = {}
        for i, tok in enumerate(tokens):
            bins.setdefault(tok, []).append(i)
        for label, items in bins.items():
            total = sum(self.weights[i] for i in items)
            if total > self.CAPACITY:
                return self._malformed(
                    f"bin '{label}' holds items {items} totalling {total} > capacity "
                    f"{self.CAPACITY}. Submission rejected; try again."
                )
        used = len(bins)
        self.done = True
        if used == self.optimal:
            reward = 1.0
        elif used == self.optimal + 1:
            reward = 0.6
        elif used == self.optimal + 2:
            reward = 0.3
        else:
            reward = 0.1
        obs = (
            f"Valid packing accepted: {used} bin(s) used (optimal was {self.optimal}). "
            f"Reward = {reward:.2f}."
        )
        return obs, reward, True, False, {"bins_used": used, "optimal": self.optimal}
