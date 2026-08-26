import random


class BanquetSeatingEnv:
    GUESTS = ["Ana", "Ben", "Cleo", "Dov", "Eli", "Fay"]
    MAX_STEPS = 9

    def __init__(self):
        self.rng = None
        self.seats = {}
        self.pos_to_guest = []
        self.step_count = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        order = list(self.GUESTS)
        self.rng.shuffle(order)
        self.pos_to_guest = order
        self.seats = {g: i for i, g in enumerate(order)}
        self.step_count = 0
        self.done = False
        obs = (
            "BANQUET SEATING DEDUCTION\n"
            "Six guests sit around a round banquet table at seats 0..5, "
            "clockwise, in a hidden order. Guests: "
            f"{', '.join(self.GUESTS)}.\n"
            "Goal: work out the clockwise seating order and submit it.\n"
            "Actions:\n"
            "  ASK <name1> <name2>  -> reply is ADJACENT (they sit next to "
            "each other), OPPOSITE (directly across the table), or APART "
            "(neither).\n"
            "  SUBMIT <n1> <n2> <n3> <n4> <n5> <n6>  -> give the full "
            "clockwise order, starting from whichever seat Ana occupies. "
            "This ends the episode.\n"
            f"You have {self.MAX_STEPS} total actions (ASK or SUBMIT) before "
            "the banquet begins without you."
        )
        return obs, {}

    def _dist(self, a, b):
        pa, pb = self.seats[a], self.seats[b]
        d = abs(pa - pb) % 6
        return min(d, 6 - d)

    def _canonical(self, order):
        i = order.index("Ana")
        return order[i:] + order[:i]

    def _score(self, names):
        true_order = self._canonical(self.pos_to_guest)
        mirror_order = self._canonical(list(reversed(self.pos_to_guest)))
        best = 0
        for target in (true_order, mirror_order):
            correct = sum(1 for x, y in zip(names, target) if x == y)
            best = max(best, correct)
        return best / 6.0

    def _check_budget(self):
        if self.step_count >= self.MAX_STEPS:
            self.done = True
            return False, True
        return False, False

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}
        self.step_count += 1
        parts = action.strip().split()

        if not parts:
            terminated, truncated = self._check_budget()
            return ("Empty action. Use ASK <n1> <n2> or SUBMIT <6 names>.",
                    0.0, terminated, truncated, {})

        cmd = parts[0].upper()

        if cmd == "ASK" and len(parts) == 3:
            a, b = parts[1], parts[2]
            if a not in self.seats or b not in self.seats or a == b:
                terminated, truncated = self._check_budget()
                return (f"Invalid guest name(s). Valid guests: "
                        f"{', '.join(self.GUESTS)}.",
                        0.0, terminated, truncated, {})
            d = self._dist(a, b)
            rel = "ADJACENT" if d == 1 else ("OPPOSITE" if d == 3 else "APART")
            terminated, truncated = self._check_budget()
            left = self.MAX_STEPS - self.step_count
            obs = f"{a} and {b} are {rel}. ({left} actions left.)"
            return obs, 0.0, terminated, truncated, {}

        if cmd == "SUBMIT" and len(parts) == 7:
            names = parts[1:]
            if sorted(names) != sorted(self.GUESTS) or names[0] != "Ana":
                terminated, truncated = self._check_budget()
                return ("Invalid SUBMIT: list all six guests exactly once, "
                        "starting with Ana.",
                        0.0, terminated, truncated, {})
            reward = self._score(names)
            self.done = True
            success = reward >= 0.999
            obs = (f"SUBMIT scored {reward:.3f} "
                   f"({'correct' if success else 'partially correct'}).")
            return obs, reward, True, False, {"success": success}

        terminated, truncated = self._check_budget()
        return ("Malformed action. Use 'ASK <name1> <name2>' or "
                "'SUBMIT <n1> ... <n6>' (six names starting with Ana).",
                0.0, terminated, truncated, {})
