import random


class RoundRobinDeductionEnv:
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.done = False
        pool = [
            "Falcons", "Hornets", "Wolves", "Rangers", "Titans", "Comets",
            "Vipers", "Bears", "Eagles", "Sharks", "Panthers", "Cobras",
        ]
        self.n = 4
        self.teams = self.rng.sample(pool, self.n)
        self.order = self.teams[:]
        self.rng.shuffle(self.order)
        self.team_lookup = {t.lower(): t for t in self.teams}
        self.best_correct = 0
        listing = sorted(self.teams)
        obs = (
            "ROUND-ROBIN DEDUCTION\n"
            f"Teams {', '.join(listing)} each played every other team exactly "
            "once (no draws). Every result came from one fixed, hidden strength "
            "order: if team P is stronger than team Q, P beat Q, and strength "
            "is transitive (a strict total order over all teams).\n"
            "Goal: determine the full strength order from strongest to weakest.\n"
            "Actions (send exactly one per turn):\n"
            "  QUERY <teamA> <teamB>  - reveals which team won their match\n"
            "  RANK <t1> <t2> <t3> <t4>  - submit your guess, strongest to weakest\n"
            "You get up to 10 turns total. A RANK guess never ends the episode "
            "unless every position is correct or turns run out, and it only "
            "rewards positions correct beyond your best guess so far.\n"
            f"Teams: {', '.join(listing)}"
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()
        tokens = text.split()
        reward = 0.0
        terminated = False

        if not tokens:
            obs = ("Empty action. Use 'QUERY <teamA> <teamB>' or "
                   "'RANK <t1> <t2> <t3> <t4>'.")
        else:
            cmd = tokens[0].upper()
            if cmd == "QUERY":
                if len(tokens) != 3:
                    obs = "Malformed QUERY. Format: QUERY <teamA> <teamB>."
                else:
                    a = self.team_lookup.get(tokens[1].lower())
                    b = self.team_lookup.get(tokens[2].lower())
                    if a is None or b is None:
                        obs = ("Unknown team name. Valid teams: "
                               f"{', '.join(sorted(self.teams))}.")
                    elif a == b:
                        obs = "QUERY needs two different teams."
                    else:
                        ia, ib = self.order.index(a), self.order.index(b)
                        winner, loser = (a, b) if ia < ib else (b, a)
                        obs = f"{winner} defeated {loser} in their round-robin match."
            elif cmd == "RANK":
                if len(tokens) != self.n + 1:
                    obs = (f"Malformed RANK. Provide exactly {self.n} team "
                           "names, strongest to weakest.")
                else:
                    guess_raw = tokens[1:]
                    resolved = [self.team_lookup.get(g.lower()) for g in guess_raw]
                    if any(r is None for r in resolved) or len(set(resolved)) != self.n:
                        obs = "RANK must name each team exactly once. Try again."
                    else:
                        correct = sum(
                            1 for i in range(self.n) if resolved[i] == self.order[i]
                        )
                        frac = correct / self.n
                        prev_frac = self.best_correct / self.n
                        reward = max(0.0, frac - prev_frac)
                        if correct > self.best_correct:
                            self.best_correct = correct
                        if correct == self.n:
                            terminated = True
                            self.done = True
                            obs = (f"Correct! True order: {', '.join(self.order)}. "
                                   f"Solved in {self.step_count} turns.")
                        else:
                            obs = (f"{correct} of {self.n} positions correct. "
                                   f"Best so far: {self.best_correct}/{self.n}.")
            else:
                obs = "Unknown command. Use QUERY or RANK."

        truncated = (not terminated) and self.step_count >= 10
        if truncated:
            self.done = True
            obs += f" Out of turns. True order was: {', '.join(self.order)}."

        return obs, reward, terminated, truncated, {}
