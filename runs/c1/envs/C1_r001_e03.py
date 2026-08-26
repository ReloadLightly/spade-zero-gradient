import random


class HeadTableSeatingEnv:
    """Deduce a hidden clockwise seating order from pairwise proximity comparisons."""

    NAMES = ["Ada", "Bram", "Cleo", "Dez", "Elin"]
    MAX_STEPS = 10

    def __init__(self):
        self._name_lookup = {n.upper(): n for n in self.NAMES}
        self.rng = None
        self.order = []
        self.step_count = 0
        self.credited = [False] * 5
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.order = self.NAMES[:]
        self.rng.shuffle(self.order)
        self.step_count = 0
        self.credited = [False] * 5
        self.done = False
        obs = (
            "HEAD TABLE SEATING DEDUCTION\n"
            "Six seats sit clockwise around a round table: seat 1 is the Host (fixed), "
            "and seats 2-6 hold five guests in a hidden order: Ada, Bram, Cleo, Dez, Elin.\n"
            "Goal: within 10 total steps, determine the exact clockwise order of seats 2-6.\n"
            "Actions (exactly one per step):\n"
            "  COMPARE <name1> <name2> -- learn which of the two guests sits closer to the "
            "Host (fewer seats clockwise from seat 1).\n"
            "  SOLVE <name1> <name2> <name3> <name4> <name5> -- submit your guess for seats "
            "2 through 6, in order.\n"
            "Each SOLVE reports how many named seats are correct and permanently banks reward "
            "for any seat you name correctly for the first time. Naming all five correctly at "
            "once ends the episode with full reward. You have 10 steps total."
        )
        return obs, {}

    def _resolve(self, token):
        return self._name_lookup.get(token.upper())

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        parts = action.strip().split()
        if not parts:
            obs = "Empty action. Use 'COMPARE <name1> <name2>' or 'SOLVE <5 names>'."
        else:
            verb = parts[0].upper()
            if verb == "COMPARE" and len(parts) == 3:
                a = self._resolve(parts[1])
                b = self._resolve(parts[2])
                if a is None or b is None or a == b:
                    obs = ("Invalid COMPARE: give two distinct guest names from "
                            "Ada, Bram, Cleo, Dez, Elin.")
                else:
                    if self.order.index(a) < self.order.index(b):
                        obs = f"{a} sits closer to the Host than {b}."
                    else:
                        obs = f"{b} sits closer to the Host than {a}."
            elif verb == "SOLVE" and len(parts) == 6:
                guess = [self._resolve(t) for t in parts[1:]]
                if None in guess or sorted(guess) != sorted(self.NAMES):
                    obs = ("Invalid SOLVE: name each of Ada, Bram, Cleo, Dez, Elin "
                            "exactly once, in seat order.")
                else:
                    correct = 0
                    newly = 0
                    for i in range(5):
                        if guess[i] == self.order[i]:
                            correct += 1
                            if not self.credited[i]:
                                self.credited[i] = True
                                newly += 1
                    reward += 0.1 * newly
                    if correct == 5:
                        reward += 0.5
                        terminated = True
                        self.done = True
                        obs = f"Correct! Seats 2-6: {', '.join(self.order)}."
                    else:
                        obs = f"{correct}/5 seats correctly named. Keep deducing the rest."
            else:
                obs = ("Malformed action. Use 'COMPARE <name1> <name2>' or "
                        "'SOLVE <name1> <name2> <name3> <name4> <name5>'.")

        if not self.done and self.step_count >= self.MAX_STEPS:
            truncated = True
            self.done = True

        return obs, reward, terminated, truncated, {}
