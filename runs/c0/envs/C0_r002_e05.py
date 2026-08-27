import random


class GridBalanceEnv:
    """Balance a fixed demand across four hidden-cost, hidden-capacity
    energy sources at minimum total cost."""

    SOURCES = ["solar", "wind", "gas", "battery"]
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.step_count = 0
        self.cost = {}
        self.cap = {}
        self.demand = 0
        self.optimal_cost = 0
        self.probed = set()
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.done = False
        self.probed = set()

        cost_vals = self.rng.sample(range(2, 10), 4)
        cap_vals = [self.rng.randint(5, 20) for _ in range(4)]
        self.cost = dict(zip(self.SOURCES, cost_vals))
        self.cap = dict(zip(self.SOURCES, cap_vals))

        order = sorted(self.SOURCES, key=lambda s: self.cost[s])
        k = self.rng.randint(2, 3)
        full_sources = order[: k - 1]
        partial_source = order[k - 1]
        demand = sum(self.cap[s] for s in full_sources)
        remainder = self.rng.randint(1, self.cap[partial_source])
        demand += remainder
        self.demand = demand
        self.optimal_cost = (
            sum(self.cost[s] * self.cap[s] for s in full_sources)
            + self.cost[partial_source] * remainder
        )

        obs = (
            "GRID DISPATCH. Demand to meet exactly: {d} units.\n"
            "Sources (name only, prices and limits hidden): "
            "solar, wind, gas, battery.\n"
            "Each source charges a constant price per unit it supplies, "
            "up to its own hidden capacity.\n"
            "Actions (send exactly one per turn):\n"
            "  PROBE <source>  -- reveals that source's price/unit and capacity.\n"
            "  COMMIT <solar> <wind> <gas> <battery>  -- four non-negative "
            "integers, in that order, giving units drawn from each source. "
            "The four amounts must sum to exactly the demand and each must "
            "not exceed that source's capacity.\n"
            "A COMMIT that fails those checks costs no reward but does not "
            "end the episode -- you may probe more and commit again.\n"
            "A COMMIT that succeeds ends the episode; reward depends on how "
            "close your total cost is to the cheapest possible plan.\n"
            "You have {m} steps total."
        ).format(d=self.demand, m=self.MAX_STEPS)
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()
        parts = text.split()

        reward = 0.0
        terminated = False
        obs = ""

        if not parts:
            obs = "Empty action. Use PROBE <source> or COMMIT <s> <w> <g> <b>."
        else:
            verb = parts[0].upper()
            if verb == "PROBE":
                if len(parts) != 2 or parts[1].lower() not in self.SOURCES:
                    obs = (
                        "Malformed PROBE. Format: PROBE <source>, source in "
                        "{solar, wind, gas, battery}."
                    )
                else:
                    src = parts[1].lower()
                    self.probed.add(src)
                    obs = "{s}: price {c}/unit, capacity {cap} units.".format(
                        s=src, c=self.cost[src], cap=self.cap[src]
                    )
            elif verb == "COMMIT":
                if len(parts) != 5:
                    obs = (
                        "Malformed COMMIT. Format: COMMIT <solar> <wind> "
                        "<gas> <battery> -- four non-negative integers."
                    )
                else:
                    try:
                        vals = [int(p) for p in parts[1:5]]
                        if any(v < 0 for v in vals):
                            raise ValueError
                    except ValueError:
                        obs = (
                            "Malformed COMMIT. All four amounts must be "
                            "non-negative integers."
                        )
                        vals = None

                    if vals is not None:
                        alloc = dict(zip(self.SOURCES, vals))
                        violations = [
                            s for s in self.SOURCES if alloc[s] > self.cap[s]
                        ]
                        total = sum(vals)
                        if violations:
                            msgs = [
                                "{s}: requested {r} exceeds capacity {c} by {d}".format(
                                    s=s,
                                    r=alloc[s],
                                    c=self.cap[s],
                                    d=alloc[s] - self.cap[s],
                                )
                                for s in violations
                            ]
                            obs = "Infeasible commit -- " + "; ".join(msgs)
                        elif total != self.demand:
                            diff = self.demand - total
                            direction = "short by" if diff > 0 else "over by"
                            obs = (
                                "Infeasible commit -- total supplied {t}, "
                                "demand is {d} ({dir} {amt}).".format(
                                    t=total,
                                    d=self.demand,
                                    dir=direction,
                                    amt=abs(diff),
                                )
                            )
                        else:
                            actual_cost = sum(
                                self.cost[s] * alloc[s] for s in self.SOURCES
                            )
                            efficiency = self.optimal_cost / actual_cost
                            reward = min(1.0, 0.3 + 0.7 * efficiency)
                            terminated = True
                            self.done = True
                            obs = (
                                "Commit accepted. Total cost {ac}, cheapest "
                                "possible plan costs {oc}. Reward {r:.3f}."
                            ).format(ac=actual_cost, oc=self.optimal_cost, r=reward)
            else:
                obs = (
                    "Unknown action verb. Use PROBE <source> or "
                    "COMMIT <solar> <wind> <gas> <battery>."
                )

        truncated = False
        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += " Step limit reached; episode ends."

        info = {"step": self.step_count, "demand": self.demand}
        return obs, reward, terminated, truncated, info
