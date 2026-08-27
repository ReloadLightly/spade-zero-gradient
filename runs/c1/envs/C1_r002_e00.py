import random
import re


class HollowFenLedgerEnv:
    NAMES = ["the Elder", "the Miller", "the Weaver", "the Smith",
              "the Baker", "the Cooper", "the Tanner"]
    ASK_BUDGET = 7
    N = 7  # villagers 0..6; villager 0 is the Elder, known to be a knight

    def __init__(self):
        self.rng = None
        self.types = None  # 1 = knight (truthful), 0 = knave (liar)
        self.asks_used = 0
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        while True:
            others = [self.rng.randint(0, 1) for _ in range(self.N - 1)]
            if 0 < sum(others) < self.N - 1:
                break
        self.types = [1] + others
        self.asks_used = 0
        self.steps = 0
        self.done = False
        roster = ", ".join(f"{i}={self.NAMES[i]}" for i in range(self.N))
        obs = (
            "You are investigating the village of Hollow Fen. Six residents "
            "(villagers 1-6) are each secretly a knight (always tells the "
            "truth) or a knave (always lies). Villager 0, the Elder, is "
            "known by public record to be a knight.\n"
            f"Roster: {roster}.\n"
            "ACTIONS:\n"
            "  ASK <i> <j>  - villager i is asked 'Is villager j a knight?' "
            "and answers 'yes' or 'no'.\n"
            "  GUESS <t1> <t2> <t3> <t4> <t5> <t6>  - submit villagers 1..6's "
            "types in order, each token K (knight) or N (knave). This ends "
            "the episode.\n"
            f"You may ASK at most {self.ASK_BUDGET} times before you must "
            "GUESS. Total steps (asks plus the guess) are capped at 10.\n"
            "Reward: 1/6 for each of the six villagers you correctly "
            "identify in your final GUESS."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}
        self.steps += 1
        action = (action or "").strip()

        ask_match = re.fullmatch(r"ASK\s+(\d+)\s+(\d+)", action, re.IGNORECASE)
        guess_match = re.fullmatch(
            r"GUESS\s+([KNkn])\s+([KNkn])\s+([KNkn])\s+([KNkn])\s+([KNkn])\s+([KNkn])",
            action, re.IGNORECASE,
        )

        if ask_match:
            i, j = int(ask_match.group(1)), int(ask_match.group(2))
            if not (0 <= i < self.N and 0 <= j < self.N):
                return self._malformed(f"Villager indices must be 0-{self.N - 1}.")
            if self.asks_used >= self.ASK_BUDGET:
                return self._malformed("No ASK actions remain; you must GUESS now.")
            self.asks_used += 1
            answer = "yes" if self.types[i] == self.types[j] else "no"
            obs = (
                f"{self.NAMES[i]} is asked 'Is villager {j} a knight?' and "
                f"answers: '{answer}'. ({self.ASK_BUDGET - self.asks_used} "
                f"asks remaining, {10 - self.steps} steps remaining.)"
            )
            if self.steps >= 10:
                self.done = True
                return obs + " No steps remain; the investigation closes.", 0.0, False, True, {}
            return obs, 0.0, False, False, {}

        if guess_match:
            guess = [1 if g.upper() == "K" else 0 for g in guess_match.groups()]
            actual = self.types[1:]
            correct = sum(1 for g, a in zip(guess, actual) if g == a)
            reward = correct / 6.0
            self.done = True
            truth = ", ".join(
                f"{self.NAMES[k]}={'knight' if t else 'knave'}"
                for k, t in enumerate(self.types) if k > 0
            )
            obs = f"You guessed {correct}/6 correctly. The truth: {truth}."
            return obs, reward, True, False, {}

        return self._malformed(
            "Malformed action. Use 'ASK <i> <j>' or "
            "'GUESS <t1> <t2> <t3> <t4> <t5> <t6>' with tokens K or N."
        )

    def _malformed(self, msg):
        if self.steps >= 10:
            self.done = True
            return msg + " No steps remain; the investigation closes.", 0.0, False, True, {}
        return msg, 0.0, False, False, {}
