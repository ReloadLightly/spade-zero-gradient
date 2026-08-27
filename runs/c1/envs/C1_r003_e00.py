import random


class HeraldsTallyEnv:
    def __init__(self):
        self.n = 6
        self.max_steps = 10
        self.rng = None
        self.hats = []
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.hats = [self.rng.choice(['G', 'S']) for _ in range(self.n)]
        self.steps = 0
        self.done = False
        obs = (
            "Six nobles stand in a line, numbered 1 to 6 left to right. Each wears a "
            "hidden hat, Gold ('G') or Silver ('S'). Every noble can see all the other "
            "nobles' hats but not their own; you can see none of them directly.\n"
            "On each turn choose ONE action:\n"
            "  ask <n> left   -- noble n truthfully states how many Gold hats they see "
            "among the nobles standing to their left (lower numbers).\n"
            "  ask <n> right  -- noble n truthfully states how many Gold hats they see "
            "among the nobles standing to their right (higher numbers).\n"
            "  guess <c1> <c2> <c3> <c4> <c5> <c6> -- submit your final answer for all "
            "six hats in order (each ci is G or S). This ends the episode immediately.\n"
            f"You have {self.max_steps} actions total. Gather reports, then guess once "
            "you are confident."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "The episode has already ended.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()
        parts = action.split()

        if len(parts) == 3 and parts[0].lower() == 'ask':
            n = None
            try:
                n = int(parts[1])
            except ValueError:
                pass
            side = parts[2].lower()
            if n is None or not (1 <= n <= self.n) or side not in ('left', 'right'):
                obs = "Malformed ask. Use: ask <1-6> left|right"
                return obs, 0.0, False, self.steps >= self.max_steps, {}
            idx = n - 1
            if side == 'left':
                count = sum(1 for h in self.hats[:idx] if h == 'G')
                obs = f"Noble {n} says: 'I see {count} Gold hat(s) to my left.'"
            else:
                count = sum(1 for h in self.hats[idx + 1:] if h == 'G')
                obs = f"Noble {n} says: 'I see {count} Gold hat(s) to my right.'"
            return obs, 0.0, False, self.steps >= self.max_steps, {}

        if len(parts) == 7 and parts[0].lower() == 'guess':
            guesses = [p.upper() for p in parts[1:]]
            if any(g not in ('G', 'S') for g in guesses):
                obs = "Malformed guess. Each of the 6 values must be G or S."
                return obs, 0.0, False, self.steps >= self.max_steps, {}
            correct = sum(1 for g, h in zip(guesses, self.hats) if g == h)
            reward = correct / self.n
            self.done = True
            obs = (
                f"Final guess recorded: {' '.join(guesses)}. You correctly identified "
                f"{correct}/{self.n} hats."
            )
            return obs, reward, True, False, {}

        obs = "Unrecognized action. Use 'ask <1-6> left|right' or 'guess <6 letters>'."
        return obs, 0.0, False, self.steps >= self.max_steps, {}
