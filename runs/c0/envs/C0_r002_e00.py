import random


class KnaveAmongKnightsEnv:
    NAME_POOL = [
        "Alden", "Brina", "Corwin", "Delia", "Esben",
        "Fenna", "Garrik", "Hesper", "Ilo", "Junne",
    ]
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.villagers = []
        self.knave = None
        self.cleared = set()
        self.step_count = 0
        self.reward_earned = 0.0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        n = self.rng.choice([4, 5, 6])
        self.villagers = self.rng.sample(self.NAME_POOL, n)
        self.knave = self.rng.choice(self.villagers)
        self.cleared = set()
        self.step_count = 0
        self.reward_earned = 0.0
        self.done = False

        names_str = ", ".join(self.villagers)
        obs = (
            "You are investigating a village of {n} inhabitants: {names}. "
            "Exactly ONE of them is a knave who always lies; every other "
            "inhabitant is a knight who always tells the truth. Find the "
            "knave.\n\n"
            "Action format (send exactly one per turn):\n"
            "  ASK <name1> <name2>  -- ask <name1> the yes/no question "
            "'Is <name2> a knight?' and receive their answer.\n"
            "  ACCUSE <name>        -- name the villager you believe is the "
            "knave. This ends the game.\n\n"
            "You have at most {steps} turns total, including your final "
            "ACCUSE. Names must be spelled exactly as listed, and <name1> "
            "and <name2> in an ASK must be two different people."
        ).format(n=n, names=names_str, steps=self.MAX_STEPS)
        info = {"villagers": list(self.villagers)}
        return obs, info

    def _find_name(self, token):
        token = token.strip().lower()
        for v in self.villagers:
            if v.lower() == token:
                return v
        return None

    def _check_step_limit(self):
        if self.step_count >= self.MAX_STEPS:
            self.done = True
            return False, True
        return False, False

    def step(self, action):
        if self.done:
            return "The investigation has already ended.", 0.0, True, False, {}

        self.step_count += 1
        action = (action or "").strip()
        parts = action.split()

        if parts and parts[0].upper() == "ACCUSE" and len(parts) == 2:
            target = self._find_name(parts[1])
            if target is None:
                obs = "'{0}' is not a villager in this case. Use one of: {1}.".format(
                    parts[1], ", ".join(self.villagers)
                )
                terminated, truncated = self._check_step_limit()
                return obs, 0.0, terminated, truncated, {}

            self.done = True
            if target == self.knave:
                reward = max(0.0, 1.0 - self.reward_earned)
                self.reward_earned += reward
                obs = "Correct! {0} was the knave. Case closed.".format(target)
                return obs, reward, True, False, {"correct": True}
            else:
                obs = (
                    "Wrong. {0} was not the knave. The real knave slips "
                    "away. Case closed."
                ).format(target)
                return obs, 0.0, True, False, {"correct": False}

        if parts and parts[0].upper() == "ASK" and len(parts) == 3:
            a = self._find_name(parts[1])
            b = self._find_name(parts[2])
            if a is None or b is None:
                bad = parts[1] if a is None else parts[2]
                obs = "'{0}' is not a villager in this case. Use one of: {1}.".format(
                    bad, ", ".join(self.villagers)
                )
                terminated, truncated = self._check_step_limit()
                return obs, 0.0, terminated, truncated, {}
            if a == b:
                obs = "Ask <name1> about a DIFFERENT villager, not themselves."
                terminated, truncated = self._check_step_limit()
                return obs, 0.0, terminated, truncated, {}

            a_is_knave = (a == self.knave)
            b_is_knave = (b == self.knave)
            same_type = (a_is_knave == b_is_knave)

            reward = 0.0
            if same_type:
                newly = [x for x in (a, b) if x not in self.cleared]
                for x in newly:
                    self.cleared.add(x)
                n_possible_knights = len(self.villagers) - 1
                if n_possible_knights > 0 and newly:
                    reward = 0.5 * len(newly) / n_possible_knights
                    self.reward_earned += reward

            obs = "{0} answers your question about {1}: \"{2}.\"".format(
                a, b, "Yes" if same_type else "No"
            )
            terminated, truncated = self._check_step_limit()
            return obs, reward, terminated, truncated, {}

        obs = (
            "Malformed action. Use 'ASK <name1> <name2>' or "
            "'ACCUSE <name>' with names from: {0}."
        ).format(", ".join(self.villagers))
        terminated, truncated = self._check_step_limit()
        return obs, 0.0, terminated, truncated, {}
