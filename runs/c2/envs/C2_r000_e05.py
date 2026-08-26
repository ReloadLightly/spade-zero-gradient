import random


class BracketSeedingEnv:
    def __init__(self):
        self.teams = ['A', 'B', 'C', 'D']
        self.max_steps = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        vals = self.rng.sample(range(10, 100), 4)
        self.strengths = dict(zip(self.teams, vals))
        self.step_count = 0
        self.done = False
        self.edges = set()
        self.milestone_max_awarded = False
        self.milestone_min_awarded = False
        obs = (
            "Seed a 4-team single-elimination bracket (2 semifinals + 1 final) "
            "to MINIMIZE the tournament's total expected number of upsets. "
            "Each team's true strength is a hidden number. An 'upset' in any "
            "game is the weaker (lower-strength) team winning that game; a "
            "team with strength s beats an opponent with strength s' with "
            "probability s/(s+s'). Teams: A, B, C, D.\n"
            "Actions (exactly one per turn):\n"
            "  COMPARE X Y   - reveals which of two teams has higher hidden strength\n"
            "  SUBMIT W X Y Z - locks in (W vs X) as semifinal 1 and (Y vs Z) as "
            "semifinal 2, then ends the episode\n"
            f"You have {self.max_steps} steps total, comparisons and submit both count."
        )
        return obs, {}

    def _closure(self):
        reach = {t: set() for t in self.teams}
        for (w, l) in self.edges:
            reach[w].add(l)
        changed = True
        while changed:
            changed = False
            for t in self.teams:
                add = set()
                for x in reach[t]:
                    for y in reach[x]:
                        if y != t and y not in reach[t]:
                            add.add(y)
                if add:
                    reach[t] |= add
                    changed = True
        return reach

    def _known_max(self, reach):
        for t in self.teams:
            if len(reach[t]) == 3:
                return t
        return None

    def _known_min(self, reach):
        for t in self.teams:
            if all(t in reach[x] for x in self.teams if x != t):
                return t
        return None

    def _expected_upsets(self, a, b, c, d):
        sa, sb, sc, sd = (self.strengths[x] for x in (a, b, c, d))
        p_a = sa / (sa + sb)
        p_b = 1 - p_a
        p_c = sc / (sc + sd)
        p_d = 1 - p_c
        upset_g1 = min(sa, sb) / (sa + sb)
        upset_g2 = min(sc, sd) / (sc + sd)
        combos = [(a, c, p_a * p_c), (a, d, p_a * p_d),
                  (b, c, p_b * p_c), (b, d, p_b * p_d)]
        upset_final = 0.0
        for f1, f2, prob in combos:
            s1, s2 = self.strengths[f1], self.strengths[f2]
            upset_final += prob * (min(s1, s2) / (s1 + s2))
        return upset_g1 + upset_g2 + upset_final

    def _ranked_partitions(self):
        a = self.teams[0]
        others = self.teams[1:]
        results = []
        for i, partner in enumerate(others):
            rest = [t for t in others if t != partner]
            w, x, y, z = a, partner, rest[0], rest[1]
            val = self._expected_upsets(w, x, y, z)
            key = frozenset([frozenset((w, x)), frozenset((y, z))])
            results.append((key, val))
        results.sort(key=lambda pair: pair[1])
        return results

    def step(self, action):
        if self.done:
            return "Episode already ended.", 0.0, True, False, {}

        self.step_count += 1
        tokens = action.strip().split()
        reward = 0.0
        terminated = False
        truncated = False
        obs = ""

        if len(tokens) == 3 and tokens[0].upper() == 'COMPARE':
            a, b = tokens[1].upper(), tokens[2].upper()
            if a not in self.teams or b not in self.teams or a == b:
                obs = "Invalid COMPARE: give two distinct team letters from A, B, C, D."
            else:
                if self.strengths[a] > self.strengths[b]:
                    winner, loser = a, b
                else:
                    winner, loser = b, a
                self.edges.add((winner, loser))
                reach = self._closure()
                parts = [f"{winner} is stronger than {loser}."]
                if not self.milestone_max_awarded:
                    mx = self._known_max(reach)
                    if mx is not None:
                        self.milestone_max_awarded = True
                        reward += 0.3
                        parts.append(f"Milestone: {mx} is now provably the strongest team (+0.3).")
                if not self.milestone_min_awarded:
                    mn = self._known_min(reach)
                    if mn is not None:
                        self.milestone_min_awarded = True
                        reward += 0.3
                        parts.append(f"Milestone: {mn} is now provably the weakest team (+0.3).")
                obs = " ".join(parts) + f" Steps used: {self.step_count}/{self.max_steps}."
        elif len(tokens) == 5 and tokens[0].upper() == 'SUBMIT':
            subs = [t.upper() for t in tokens[1:]]
            if sorted(subs) != sorted(self.teams):
                obs = "Invalid SUBMIT: list all four teams A, B, C, D exactly once, e.g. SUBMIT A D B C."
            else:
                key = frozenset([frozenset(subs[0:2]), frozenset(subs[2:4])])
                value = self._expected_upsets(subs[0], subs[1], subs[2], subs[3])
                ranked = self._ranked_partitions()
                rank_index = next(i for i, (k, v) in enumerate(ranked) if k == key)
                if rank_index == 0:
                    submit_reward, verdict = 0.4, "optimal"
                elif rank_index == 1:
                    submit_reward, verdict = 0.15, "suboptimal"
                else:
                    submit_reward, verdict = 0.0, "poor"
                reward = submit_reward
                terminated = True
                self.done = True
                obs = (f"Bracket locked: ({subs[0]} vs {subs[1]}) and ({subs[2]} vs {subs[3]}). "
                       f"This pairing is {verdict} (expected upsets {value:.3f}, "
                       f"best possible {ranked[0][1]:.3f}). Reward this step: {submit_reward:.2f}.")
        else:
            obs = "Malformed action. Use 'COMPARE X Y' or 'SUBMIT W X Y Z' with team letters A, B, C, D."

        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            self.done = True
            obs += " Step limit reached without a SUBMIT; episode ends."

        return obs, reward, terminated, truncated, {}
