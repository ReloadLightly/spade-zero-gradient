import random
import re


class GridBalanceEnv:
    SOURCES = ["Solar", "Wind", "Hydro", "Gas", "Coal"]
    MAX_INSPECTS = 3
    STEP_LIMIT = 10

    def __init__(self):
        self.rng = None
        self.capacities = {}
        self.costs = {}
        self.tiers = {}
        self.demand = 0
        self.optimal_cost = 0
        self.worst_cost = 0
        self.inspect_count = 0
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.capacities = {s: self.rng.randint(4, 10) for s in self.SOURCES}
        self.costs = {s: self.rng.randint(2, 9) for s in self.SOURCES}
        total_cap = sum(self.capacities.values())
        self.demand = max(6, round(total_cap * 0.6))
        self.demand = min(self.demand, total_cap - 1)

        def tier(c):
            if c <= 4:
                return "low"
            elif c <= 6:
                return "mid"
            else:
                return "high"

        self.tiers = {s: tier(self.costs[s]) for s in self.SOURCES}
        self.optimal_cost = self._merit_order_cost(self.demand, reverse=False)
        self.worst_cost = self._merit_order_cost(self.demand, reverse=True)

        self.inspect_count = 0
        self.steps = 0
        self.done = False

        lines = [
            "GRID BALANCE CONSOLE",
            "Goal: dispatch exactly " + str(self.demand) + " MWh total across the " + str(len(self.SOURCES)) +
            " sources below, minimizing total cost. Each source has a known capacity (max MWh) and a HIDDEN "
            "cost per MWh; you only see a rough cost tier for each.",
            "",
            "Sources (name: capacity MWh, cost tier):",
        ]
        for s in self.SOURCES:
            lines.append("  " + s + ": capacity=" + str(self.capacities[s]) + ", tier=" + self.tiers[s])
        lines.append("")
        lines.append("You may INSPECT at most " + str(self.MAX_INSPECTS) +
                      " sources (across the whole episode) to learn their exact cost.")
        lines.append("Actions (one per step):")
        lines.append("  inspect <Source>                     - reveal exact cost/MWh for one source")
        lines.append("  dispatch <S1>=<n1>, <S2>=<n2>, ...    - final allocation: nonnegative integer amounts, "
                      "each <= that source's capacity, summing to exactly the demand. A VALID dispatch ends the "
                      "episode; an invalid one is rejected with no penalty and does not end the episode.")
        lines.append("Cost tiers: low=2-4, mid=5-6, high=7-9 per MWh. You have " + str(self.STEP_LIMIT) +
                      " steps total.")
        return "\n".join(lines), {}

    def _merit_order_cost(self, demand, reverse=False):
        order = sorted(self.SOURCES, key=lambda s: self.costs[s], reverse=reverse)
        remaining = demand
        cost = 0
        for s in order:
            take = min(self.capacities[s], remaining)
            cost += take * self.costs[s]
            remaining -= take
            if remaining <= 0:
                break
        return cost

    def _maybe_truncate(self, obs, reward):
        if self.steps >= self.STEP_LIMIT:
            self.done = True
            return obs + " Step limit reached; episode truncated with no dispatch submitted.", reward, False, True, {}
        return obs, reward, False, False, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()

        m_inspect = re.match(r"(?i)^inspect\s+([A-Za-z]+)\s*$", action)
        m_dispatch = re.match(r"(?i)^dispatch\s+(.+)$", action)

        if m_inspect:
            name = m_inspect.group(1).strip().capitalize()
            if name not in self.SOURCES:
                obs = "Unknown source '" + name + "'. Valid sources: " + ", ".join(self.SOURCES) + "."
                return self._maybe_truncate(obs, 0.0)
            if self.inspect_count >= self.MAX_INSPECTS:
                obs = "No inspections remaining (used " + str(self.inspect_count) + "/" + str(self.MAX_INSPECTS) + ")."
                return self._maybe_truncate(obs, 0.0)
            self.inspect_count += 1
            obs = (name + ": exact cost = " + str(self.costs[name]) + " per MWh. Inspections used: " +
                   str(self.inspect_count) + "/" + str(self.MAX_INSPECTS) + ".")
            return self._maybe_truncate(obs, 0.0)

        if m_dispatch:
            body = m_dispatch.group(1)
            alloc = {}
            valid = True
            for part in body.split(","):
                part = part.strip()
                if not part:
                    continue
                mm = re.match(r"(?i)^([A-Za-z]+)\s*=\s*(-?\d+)\s*$", part)
                if not mm:
                    valid = False
                    break
                name = mm.group(1).capitalize()
                amt = int(mm.group(2))
                if name not in self.SOURCES or amt < 0:
                    valid = False
                    break
                alloc[name] = alloc.get(name, 0) + amt

            if not valid:
                obs = ("Malformed dispatch. Use: dispatch Source1=amount1, Source2=amount2, ... "
                       "with nonnegative integer amounts.")
                return self._maybe_truncate(obs, 0.0)

            for name, amt in alloc.items():
                if amt > self.capacities[name]:
                    obs = "Dispatch rejected: " + name + " allocation " + str(amt) + " exceeds capacity " + \
                          str(self.capacities[name]) + "."
                    return self._maybe_truncate(obs, 0.0)

            total = sum(alloc.values())
            if total != self.demand:
                obs = "Dispatch rejected: total " + str(total) + " MWh does not equal demand " + \
                      str(self.demand) + " MWh."
                return self._maybe_truncate(obs, 0.0)

            achieved_cost = sum(alloc.get(s, 0) * self.costs[s] for s in self.SOURCES)
            self.done = True
            span = max(self.worst_cost - self.optimal_cost, 1)
            efficiency = 1.0 - (achieved_cost - self.optimal_cost) / span
            efficiency = max(0.0, min(1.0, efficiency))
            reward = 0.3 + 0.7 * efficiency
            reward = max(0.0, min(1.0, reward))
            obs = ("Dispatch accepted. Total cost = " + str(achieved_cost) + " (optimal = " +
                   str(self.optimal_cost) + ", worst feasible = " + str(self.worst_cost) +
                   "). Demand met exactly. Episode complete, reward=" + format(reward, ".2f") + ".")
            return obs, reward, True, False, {"achieved_cost": achieved_cost, "optimal_cost": self.optimal_cost}

        obs = "Unrecognized action. Use 'inspect <Source>' or 'dispatch <Source>=<amount>, ...'."
        return self._maybe_truncate(obs, 0.0)
