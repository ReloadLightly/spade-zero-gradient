import random


class BracketSeedingEnv:
    TEAMS = ["A", "B", "C", "D", "E", "F", "G", "H"]

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.teams = list(self.TEAMS)
        values = list(range(1, 9))
        self.rng.shuffle(values)
        self.strength = dict(zip(self.teams, values))
        self.rank_order = sorted(self.teams, key=lambda t: -self.strength[t])
        self.canonical_slot_rank = self._seed_order(8)
        self.step_count = 0
        self.max_steps = 10
        self.done = False
        obs = self._opening_observation()
        info = {"teams": list(self.teams), "max_steps": self.max_steps}
        return obs, info

    def _seed_order(self, n):
        order = [1]
        size = 1
        while size < n:
            new_order = []
            for s in order:
                new_order.append(s)
                new_order.append(size * 2 + 1 - s)
            order = new_order
            size *= 2
        return order

    def _opening_observation(self):
        return (
            "You are seeding an 8-team single-elimination bracket: teams "
            f"{', '.join(self.teams)}. Bracket slots are numbered 0-7. "
            "Round 1 pairs are (0v1),(2v3),(4v5),(6v7). The Round-1 winners "
            "from slots 0-3 meet in a Round-2 match, and separately the "
            "Round-1 winners from slots 4-7 meet in a Round-2 match. Those "
            "two Round-2 winners meet in the Final. Every team has a hidden "
            "true strength. Two strong teams meeting early creates upset "
            "risk that a good seeding should avoid.\n\n"
            "Your goal: assign all 8 teams to slots 0-7 to minimize the "
            "risk of strong teams eliminating each other before the Final.\n\n"
            "Actions (send exactly one per turn):\n"
            "  SCOUT <team> <team> - watch a practice match between two "
            "teams; you learn which one is stronger.\n"
            "  SEED <t0> <t1> <t2> <t3> <t4> <t5> <t6> <t7> - final "
            "assignment of teams to slots 0..7, in order. This ends the "
            "episode.\n"
            f"You have {self.max_steps} total actions; SCOUT and SEED both "
            "count toward the limit. SEED is your one final submission, so "
            "use SCOUT first to gather evidence."
        )

    def step(self, action):
        if self.done:
            return ("Episode already ended.", 0.0, True, False, {})

        self.step_count += 1
        tokens = action.strip().split() if action else []
        reward = 0.0
        terminated = False
        obs = ""

        if not tokens:
            obs = "Empty action. Use 'SCOUT <team> <team>' or 'SEED <8 teams>'."
        else:
            cmd = tokens[0].upper()
            if cmd == "SCOUT":
                obs = self._do_scout(tokens)
            elif cmd == "SEED":
                obs, reward, terminated = self._do_seed(tokens)
            else:
                obs = "Unknown action. Use 'SCOUT <team> <team>' or 'SEED <8 teams>'."

        truncated = (not terminated) and (self.step_count >= self.max_steps)
        self.done = terminated or truncated
        if truncated and not terminated:
            obs += " No SEED was submitted in time; episode truncated with no reward."
        info = {"step": self.step_count, "remaining_steps": max(0, self.max_steps - self.step_count)}
        return obs, reward, terminated, truncated, info

    def _do_scout(self, tokens):
        if len(tokens) != 3:
            return "SCOUT requires exactly two team letters, e.g. 'SCOUT A B'."
        t1, t2 = tokens[1].upper(), tokens[2].upper()
        if t1 not in self.teams or t2 not in self.teams or t1 == t2:
            return f"Invalid team names. Choose two distinct teams from {', '.join(self.teams)}."
        if self.strength[t1] > self.strength[t2]:
            winner, loser = t1, t2
        else:
            winner, loser = t2, t1
        return f"Scouting result: Team {winner} defeated Team {loser} in a practice match ({winner} is confirmed stronger)."

    def _do_seed(self, tokens):
        if len(tokens) != 9:
            return ("SEED requires exactly 8 team letters, one per slot 0..7.", 0.0, False)
        perm = [x.upper() for x in tokens[1:]]
        if sorted(perm) != sorted(self.teams):
            return ("SEED must be a permutation of all 8 teams with no repeats.", 0.0, False)

        slot_of = {team: idx for idx, team in enumerate(perm)}
        top2 = self.rank_order[:2]
        halves = {(0 if slot_of[t] < 4 else 1) for t in top2}
        score1 = 0.3 if len(halves) == 2 else 0.0

        top4 = self.rank_order[:4]
        quarters = {slot_of[t] // 2 for t in top4}
        score2 = 0.3 if len(quarters) == 4 else 0.0

        rank_of = {team: i + 1 for i, team in enumerate(self.rank_order)}
        matches = sum(
            1 for slot in range(8)
            if rank_of[perm[slot]] == self.canonical_slot_rank[slot]
        )
        score3 = 0.4 * matches / 8

        reward = score1 + score2 + score3
        obs = (
            f"Final seeding submitted: {' '.join(perm)}.\n"
            f"Top-2 kept in separate halves: {'yes' if score1 else 'no'} (+{score1:.2f})\n"
            f"Top-4 spread across four quarters: {'yes' if score2 else 'no'} (+{score2:.2f})\n"
            f"Slots matching the strength-optimal order: {matches}/8 (+{score3:.2f})\n"
            f"Total reward: {reward:.2f}"
        )
        return obs, reward, True


BracketSeedingEnv.__name__ = "BracketSeedingEnv"
