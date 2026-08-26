import random
import itertools


class EinsteinHousesEnv:
    PETS = ['cat', 'dog', 'bird', 'fish']
    DRINKS = ['tea', 'coffee', 'juice', 'water']
    N = 4
    MAX_STEPS = 10
    HOUSE_REWARD = 0.25

    def __init__(self):
        self.rng = None
        self.pp = None
        self.dp = None
        self.clues = []
        self.clue_ptr = 0
        self.step_count = 0
        self.confirmed = [False] * self.N
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.pp = list(self.PETS)
        self.rng.shuffle(self.pp)
        self.dp = list(self.DRINKS)
        self.rng.shuffle(self.dp)
        self.clues = self._build_clues()
        self.clue_ptr = 1 if self.clues else 0
        self.step_count = 0
        self.confirmed = [False] * self.N
        self.done = False

        first_clue = self.clues[0][1] if self.clues else "No clues generated."
        obs = (
            "Four houses stand in a row, numbered 1 (left) to 4 (right). "
            "Each house has exactly one pet from {cat, dog, bird, fish} and "
            "one drink from {tea, coffee, juice, water}; every pet and every "
            "drink is used exactly once across the row.\n"
            "Goal: determine the pet and drink in every house.\n"
            f"Starting clue: {first_clue}\n"
            "Actions (send exactly one per turn):\n"
            "  CLUE                                          - reveal one more clue\n"
            "  GUESS <house 1-4> <pet> <drink>                - test one house\n"
            "  SOLVE <p1> <d1> <p2> <d2> <p3> <d3> <p4> <d4>  - submit the full row, "
            "houses 1 to 4 left to right\n"
            f"You have {self.MAX_STEPS} steps total. Reward is {self.HOUSE_REWARD} for "
            "each house you correctly pin down (via GUESS or SOLVE, awarded once per "
            "house). The episode ends once all four houses are pinned down, or when "
            "steps run out."
        )
        info = {"step": 0, "clues_remaining": max(0, len(self.clues) - self.clue_ptr)}
        return obs, info

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {"step": self.step_count}

        self.step_count += 1
        action = (action or "").strip()
        parts = action.split()
        reward = 0.0

        if not parts:
            obs = "Malformed action. Use CLUE, GUESS <house> <pet> <drink>, or SOLVE <8 tokens>."
        else:
            cmd = parts[0].upper()
            if cmd == 'CLUE':
                if self.clue_ptr < len(self.clues):
                    obs = f"Clue: {self.clues[self.clue_ptr][1]}"
                    self.clue_ptr += 1
                else:
                    obs = "No more clues available."
            elif cmd == 'GUESS' and len(parts) == 4:
                obs, reward = self._handle_guess(parts)
            elif cmd == 'SOLVE' and len(parts) == 9:
                obs, reward = self._handle_solve(parts)
            else:
                obs = "Malformed action. Use CLUE, GUESS <house> <pet> <drink>, or SOLVE <8 tokens>."

        terminated = False
        if all(self.confirmed):
            terminated = True
            self.done = True
            obs += " All houses confirmed -- solved!"

        truncated = False
        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated = True
            self.done = True

        info = {
            "step": self.step_count,
            "confirmed": list(self.confirmed),
            "clues_remaining": max(0, len(self.clues) - self.clue_ptr),
        }
        return obs, reward, terminated, truncated, info

    def _handle_guess(self, parts):
        _, house_s, pet, drink = parts
        pet = pet.lower()
        drink = drink.lower()
        if not house_s.isdigit() or not (1 <= int(house_s) <= self.N):
            return f"Malformed GUESS: house must be 1-{self.N}.", 0.0
        house = int(house_s)
        if pet not in self.PETS or drink not in self.DRINKS:
            return "Malformed GUESS: unknown pet or drink name.", 0.0
        pet_ok = self.pp[house - 1] == pet
        drink_ok = self.dp[house - 1] == drink
        reward = 0.0
        if pet_ok and drink_ok and not self.confirmed[house - 1]:
            self.confirmed[house - 1] = True
            reward = self.HOUSE_REWARD
        obs = (f"House {house}: pet {'correct' if pet_ok else 'wrong'}, "
               f"drink {'correct' if drink_ok else 'wrong'}.")
        return obs, reward

    def _handle_solve(self, parts):
        tokens = [p.lower() for p in parts[1:]]
        houses = [(tokens[2 * i], tokens[2 * i + 1]) for i in range(self.N)]
        for pet, drink in houses:
            if pet not in self.PETS or drink not in self.DRINKS:
                return "Malformed SOLVE: unknown pet or drink name.", 0.0
        reward = 0.0
        n_correct = 0
        for h in range(self.N):
            pet, drink = houses[h]
            correct = (self.pp[h] == pet and self.dp[h] == drink)
            if correct:
                n_correct += 1
                if not self.confirmed[h]:
                    self.confirmed[h] = True
                    reward += self.HOUSE_REWARD
        obs = f"SOLVE result: {n_correct}/{self.N} houses fully correct."
        return obs, reward

    def _build_clues(self):
        pp, dp, rng = self.pp, self.dp, self.rng
        candidates = []
        seen = set()

        def add(c):
            if c not in seen:
                seen.add(c)
                candidates.append(c)

        for pet in self.PETS:
            h = pp.index(pet) + 1
            add(('pos_pet', pet, h))
            others = [x for x in range(1, self.N + 1) if x != h]
            add(('not_pos_pet', pet, rng.choice(others)))
        for drink in self.DRINKS:
            h = dp.index(drink) + 1
            add(('pos_drink', drink, h))
        for h in range(1, self.N):
            add(('left_pet_pet', pp[h - 1], pp[h]))
            add(('left_drink_pet', dp[h - 1], pp[h]))
        for h in range(1, self.N + 1):
            add(('same_house', pp[h - 1], dp[h - 1]))
        add(('end_pet', pp[0], 'left'))
        add(('end_pet', pp[-1], 'right'))

        rng.shuffle(candidates)

        selected = []
        for c in candidates:
            selected.append(c)
            if self._unique_solution(selected) is not None:
                break
        return [(c, self._clue_text(c)) for c in selected]

    def _unique_solution(self, clues):
        sol = None
        count = 0
        for pp in itertools.permutations(self.PETS):
            for dp in itertools.permutations(self.DRINKS):
                if all(self._evaluate(c, pp, dp) for c in clues):
                    count += 1
                    if count > 1:
                        return None
                    sol = (pp, dp)
        return sol

    @staticmethod
    def _evaluate(clue, pp, dp):
        t = clue[0]
        if t == 'pos_pet':
            return pp[clue[2] - 1] == clue[1]
        if t == 'pos_drink':
            return dp[clue[2] - 1] == clue[1]
        if t == 'not_pos_pet':
            return pp[clue[2] - 1] != clue[1]
        if t == 'left_pet_pet':
            return pp.index(clue[1]) + 1 == pp.index(clue[2])
        if t == 'left_drink_pet':
            return dp.index(clue[1]) + 1 == pp.index(clue[2])
        if t == 'same_house':
            return pp.index(clue[1]) == dp.index(clue[2])
        if t == 'end_pet':
            return (pp.index(clue[1]) == 0) if clue[2] == 'left' else (pp.index(clue[1]) == len(pp) - 1)
        return False

    @staticmethod
    def _clue_text(clue):
        t = clue[0]
        if t == 'pos_pet':
            return f"The house with the {clue[1]} is house {clue[2]}."
        if t == 'pos_drink':
            return f"The house that drinks {clue[1]} is house {clue[2]}."
        if t == 'not_pos_pet':
            return f"The {clue[1]} is not in house {clue[2]}."
        if t == 'left_pet_pet':
            return f"The {clue[1]} is immediately left of the {clue[2]}."
        if t == 'left_drink_pet':
            return f"The house that drinks {clue[1]} is immediately left of the house with the {clue[2]}."
        if t == 'same_house':
            return f"The {clue[1]} and the {clue[2]} are in the same house."
        if t == 'end_pet':
            side = 'leftmost' if clue[2] == 'left' else 'rightmost'
            return f"The {clue[1]} is in the {side} house."
        return "Unknown clue."
