import random


class TruthboundChainEnv:
    """Knights-and-knaves relay: deduce every villager's type via chained
    same/different testimony, anchored to one villager of known honesty."""

    def __init__(self):
        self.names = ["Aldric", "Bryn", "Cass", "Dorin", "Elva", "Finn"]
        self.n = len(self.names)
        self.max_steps = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.types = [True]
        for _ in range(1, self.n):
            self.types.append(self.rng.choice([True, False]))
        self.step_count = 0
        self.done = False
        self.edges = {}
        self.determined = {0: True}
        self.milestone_awarded = {0}
        obs = self._intro()
        return obs, {}

    def _intro(self):
        names_line = ", ".join(f"{i}:{n}" for i, n in enumerate(self.names))
        return (
            "You are investigating a village where each inhabitant is either a "
            "Knight (always tells the truth) or a Knave (always lies).\n"
            f"Villagers: {names_line}.\n"
            f"It is publicly known that villager 0 ({self.names[0]}) is a Knight. "
            "The types of the other 5 villagers are secret.\n"
            "Action format:\n"
            "  ASK <asker_id> <about_id> -- ask villager <asker_id> whether "
            "villager <about_id> is a Knight; they answer as their true nature "
            "dictates.\n"
            "  SUBMIT <6 letters> -- one letter per villager 0-5 in order, K for "
            "Knight or N for Knave (e.g. a string of six K/N characters); ends "
            "the episode.\n"
            f"You have {self.max_steps} actions total. Deduce every villager's type."
        )

    def _claim(self, asker, about):
        if asker == about:
            return True
        if self.types[asker]:
            return self.types[about]
        return not self.types[about]

    def _propagate(self):
        newly = []
        changed = True
        while changed:
            changed = False
            for i in list(self.determined.keys()):
                for (j, same) in self.edges.get(i, []):
                    if j not in self.determined:
                        self.determined[j] = self.determined[i] if same else (
                            not self.determined[i]
                        )
                        if j not in self.milestone_awarded:
                            self.milestone_awarded.add(j)
                            newly.append(j)
                        changed = True
        return newly

    def step(self, action):
        if self.done:
            return "The investigation has already ended.", 0.0, True, False, {}

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        parts = (action or "").strip().split()

        if len(parts) == 3 and parts[0].upper() == "ASK":
            asker_s, about_s = parts[1], parts[2]
            if not (asker_s.isdigit() and about_s.isdigit()):
                obs = "Malformed ASK: ids must be numbers 0-5. Try again."
            else:
                asker, about = int(asker_s), int(about_s)
                if not (0 <= asker < self.n and 0 <= about < self.n):
                    obs = f"Malformed ASK: ids must be in range 0-{self.n - 1}. Try again."
                else:
                    claim_knight = self._claim(asker, about)
                    verb = "a Knight" if claim_knight else "a Knave"
                    obs = f'{self.names[asker]} says: "{self.names[about]} is {verb}."'
                    if asker != about:
                        same = claim_knight
                        self.edges.setdefault(asker, []).append((about, same))
                        self.edges.setdefault(about, []).append((asker, same))
                        newly = self._propagate()
                        if newly:
                            reward += 0.1 * len(newly)
                            who = ", ".join(self.names[i] for i in newly)
                            obs += f" (You can now pin down the type of: {who}.)"
        elif len(parts) == 2 and parts[0].upper() == "SUBMIT":
            guess = parts[1].upper()
            if len(guess) != self.n or any(c not in "KN" for c in guess):
                obs = f"Malformed SUBMIT: provide exactly {self.n} letters, each K or N."
            else:
                correct = sum(
                    1 for i in range(self.n) if (guess[i] == "K") == self.types[i]
                )
                reward += 0.5 * (correct / self.n)
                terminated = True
                if correct == self.n:
                    obs = f"Submission recorded: {correct}/{self.n} correct. All villagers correctly identified!"
                else:
                    obs = f"Submission recorded: {correct}/{self.n} correct. The investigation ends."
        else:
            obs = "Malformed action. Use 'ASK <id> <id>' or 'SUBMIT <letters>'."

        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            obs += " No actions remain; the investigation ends unresolved."

        if terminated or truncated:
            self.done = True

        return obs, reward, terminated, truncated, {}
