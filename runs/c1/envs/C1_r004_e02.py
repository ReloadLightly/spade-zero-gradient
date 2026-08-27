import re
import random


class BoulevardSyncDialEnv:
    """Optimize a hidden-peak piecewise-linear traffic flow function."""

    MAX_O = 19
    BASE = 100
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.peak = None
        self.left_slope = None
        self.right_slope = None
        self.steps = 0
        self.done = False

    def _score(self, o):
        if o <= self.peak:
            return self.BASE - self.left_slope * (self.peak - o)
        return self.BASE - self.right_slope * (o - self.peak)

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.peak = self.rng.randint(3, 16)
        self.left_slope = self.rng.randint(2, 5)
        self.right_slope = self.rng.randint(2, 5)
        self.steps = 0
        self.done = False

        obs = (
            "SIGNAL BANK OPTIMIZATION\n"
            "A secondary bank of traffic signals along the avenue can be shifted by a "
            "phase offset O, an integer from 0 to 19. Exactly one offset O* gives the "
            "maximum flow score of 100 vehicles/hour. Moving away from O* in either "
            "direction, the flow score drops at a constant rate per step of O, but that "
            "rate can be different depending on which direction you move.\n"
            "Your goal: determine O* and lock it in.\n"
            "Actions (send exactly one per turn):\n"
            "  EVALUATE <O>   -- reports the flow score (0-100) at offset O. Costs one action.\n"
            "  SUBMIT <O>     -- locks in O as your final answer and ends the episode.\n"
            f"You have at most {self.MAX_STEPS} actions total (evaluations and the final "
            "submit combined). Reply with one action in exactly that format."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        match = re.match(r"^\s*(EVALUATE|SUBMIT)\s+(-?\d+)\s*$", action.strip(), re.IGNORECASE)

        if not match:
            obs = (
                "Malformed action. Use exactly 'EVALUATE <O>' or 'SUBMIT <O>' with an "
                "integer offset, e.g. EVALUATE 10."
            )
            truncated = self.steps >= self.MAX_STEPS
            if truncated:
                self.done = True
            return obs, 0.0, False, truncated, {}

        verb = match.group(1).upper()
        value = int(match.group(2))

        if value < 0 or value > self.MAX_O:
            obs = f"Offset out of range. O must be an integer from 0 to {self.MAX_O}."
            truncated = self.steps >= self.MAX_STEPS
            if truncated:
                self.done = True
            return obs, 0.0, False, truncated, {}

        if verb == "EVALUATE":
            s = self._score(value)
            remaining = self.MAX_STEPS - self.steps
            obs = f"Flow score at O={value} is {s}. Actions remaining: {remaining}."
            truncated = self.steps >= self.MAX_STEPS
            if truncated:
                self.done = True
                obs += " No actions remain; episode truncated without a submission."
            return obs, 0.0, False, truncated, {}

        # SUBMIT
        self.done = True
        dist = abs(value - self.peak)
        if dist == 0:
            reward = 1.0
            verdict = "Optimal! You found the true peak offset."
        elif dist <= 1:
            reward = 0.6
            verdict = f"Close: off by {dist} from the true peak."
        elif dist <= 3:
            reward = 0.3
            verdict = f"Partial credit: off by {dist} from the true peak."
        else:
            reward = 0.0
            verdict = f"Off target: off by {dist} from the true peak."

        obs = f"Submitted O={value}. {verdict} True peak offset was {self.peak}."
        info = {
            "true_peak": self.peak,
            "left_slope": self.left_slope,
            "right_slope": self.right_slope,
            "submitted": value,
        }
        return obs, reward, True, False, info
