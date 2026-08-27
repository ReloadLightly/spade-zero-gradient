import itertools
import random


class CaravanAssignmentEnv:
    TERRAINS = ("dune", "rock", "salt")

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.n = 3
        self.team_affinity = [self.rng.choice(self.TERRAINS) for _ in range(self.n)]
        self.leg_terrain = [self.rng.choice(self.TERRAINS) for _ in range(self.n)]

        self.cost = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                base = self.rng.randint(4, 14)
                if self.team_affinity[i] == self.leg_terrain[j]:
                    base = max(2, base - self.rng.randint(3, 5))
                self.cost[i][j] = base

        self.scout_budget = 5
        self.scouts_used = 0
        self.steps = 0
        self.max_steps = 10
        self.done = False

        perm_costs = [
            sum(self.cost[i][perm[i]] for i in range(self.n))
            for perm in itertools.permutations(range(self.n))
        ]
        self.optimal_cost = min(perm_costs)
        self.worst_cost = max(perm_costs)
        self.naive_cost = sum(self.cost[i][i] for i in range(self.n))

        obs = self._render_intro()
        return obs, {}

    def _render_intro(self):
        lines = []
        lines.append(
            "CARAVAN DISPATCH: assign 3 camel teams to 3 desert legs (a one-to-one "
            "pairing) to minimize TOTAL fatigue cost. You have at most 10 actions total."
        )
        lines.append(
            "Exact pairing costs are hidden. You may spend up to 5 scouting actions "
            "to reveal exact costs before making one final, irreversible assignment."
        )
        lines.append("ACTIONS:")
        lines.append(
            "  'SCOUT <team> <leg>' — team and leg are each 1-3; reveals the exact "
            "fatigue cost of pairing that team with that leg (uses one of 5 scouts)."
        )
        lines.append(
            "  'ASSIGN <leg1> <leg2> <leg3>' — leg for team 1, team 2, team 3 "
            "respectively, must be a permutation of 1,2,3. Ends the episode."
        )
        lines.append("Public terrain info (affinity-matched pairs tend cheaper, but not always):")
        for i in range(self.n):
            lines.append(f"  Team {i + 1} favors {self.team_affinity[i]} terrain.")
        for j in range(self.n):
            lines.append(f"  Leg {j + 1} crosses {self.leg_terrain[j]} terrain.")
        lines.append(f"Scouts remaining: {self.scout_budget - self.scouts_used}. Steps used: {self.steps}/{self.max_steps}.")
        return "\n".join(lines)

    def _status_line(self):
        return (
            f"Scouts remaining: {self.scout_budget - self.scouts_used}. "
            f"Steps used: {self.steps}/{self.max_steps}."
        )

    def step(self, action):
        if self.done:
            return self._status_line(), 0.0, True, False, {}

        self.steps += 1
        text = (action or "").strip()
        parts = text.split()
        reward = 0.0
        terminated = False

        if parts and parts[0].upper() == "SCOUT" and len(parts) == 3:
            try:
                t = int(parts[1])
                l = int(parts[2])
            except ValueError:
                t = l = None
            if t is None or not (1 <= t <= self.n) or not (1 <= l <= self.n):
                obs = "Malformed SCOUT: team and leg must each be integers 1-3. " + self._status_line()
            elif self.scouts_used >= self.scout_budget:
                obs = "No scouts remaining. Submit ASSIGN when ready. " + self._status_line()
            else:
                self.scouts_used += 1
                c = self.cost[t - 1][l - 1]
                obs = f"Scouted team {t} x leg {l}: fatigue cost = {c}. " + self._status_line()
        elif parts and parts[0].upper() == "ASSIGN" and len(parts) == 4:
            try:
                legs = [int(x) for x in parts[1:]]
            except ValueError:
                legs = None
            if legs is None or sorted(legs) != list(range(1, self.n + 1)):
                obs = (
                    "Malformed ASSIGN: give exactly 3 numbers that are a permutation "
                    "of 1,2,3. " + self._status_line()
                )
            else:
                submitted_cost = sum(self.cost[i][legs[i] - 1] for i in range(self.n))
                span = self.worst_cost - self.optimal_cost
                gap = submitted_cost - self.optimal_cost
                r_valid = 0.2
                r_beats_naive = 0.2 if submitted_cost <= self.naive_cost else 0.0
                if span > 0:
                    r_close = 0.6 * max(0.0, 1.0 - gap / span)
                else:
                    r_close = 0.6
                reward = r_valid + r_beats_naive + r_close
                terminated = True
                self.done = True
                obs = (
                    f"Assignment submitted: {legs}. Total fatigue cost = {submitted_cost} "
                    f"(optimal was {self.optimal_cost}). Episode ended."
                )
        else:
            obs = (
                "Malformed action. Use 'SCOUT <team> <leg>' or "
                "'ASSIGN <leg1> <leg2> <leg3>'. " + self._status_line()
            )

        truncated = False
        if not terminated and self.steps >= self.max_steps:
            truncated = True
            self.done = True
            obs = obs + " Step budget exhausted without a final assignment."

        return obs, reward, terminated, truncated, {}
