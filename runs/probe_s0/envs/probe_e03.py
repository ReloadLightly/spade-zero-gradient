import random


class EinsteinGridDeductionEnv:
    PET_POOL = ['cat', 'dog', 'bird', 'fish', 'rabbit', 'turtle']
    DRINK_POOL = ['tea', 'coffee', 'juice', 'water', 'milk', 'cocoa']
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.pets = {}
        self.drinks = {}
        self.pets_used = []
        self.drinks_used = []
        self.clue_sequence = []
        self.clue_index = 0
        self.step_count = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.pets_used = self.rng.sample(self.PET_POOL, 3)
        self.drinks_used = self.rng.sample(self.DRINK_POOL, 3)
        self.pets = {1: self.pets_used[0], 2: self.pets_used[1], 3: self.pets_used[2]}
        self.drinks = {1: self.drinks_used[0], 2: self.drinks_used[1], 3: self.drinks_used[2]}

        self._build_clues()

        self.clue_index = 0
        self.step_count = 0
        self.done = False

        obs = (
            "Deduce the PET and DRINK for each of 3 houses (numbered 1-3, left to right). "
            "Each of 3 pets and each of 3 drinks is used exactly once, one per house.\n"
            f"Pets in play: {', '.join(self.pets_used)}.\n"
            f"Drinks in play: {', '.join(self.drinks_used)}.\n"
            "Each turn, send exactly one action:\n"
            "  CLUE  -- reveal the next true clue about the solution\n"
            "  SOLVE <pet1> <drink1> <pet2> <drink2> <pet3> <drink3>  -- your final answer "
            "for houses 1, 2, 3 in order (SOLVE ends the episode immediately)\n"
            f"You have {self.MAX_STEPS} turns total."
        )
        info = {'pets_used': list(self.pets_used), 'drinks_used': list(self.drinks_used)}
        return obs, info

    def _build_clues(self):
        houses = [1, 2, 3]
        pet_houses = self.rng.sample(houses, 2)
        link_house = self.rng.choice(pet_houses)
        remaining_for_drink = [h for h in houses if h != link_house]
        drink_direct_house = self.rng.choice(remaining_for_drink)

        clues = []
        for h in pet_houses:
            clues.append(f"House {h} has the {self.pets[h]}.")
        clues.append(
            f"The house with the {self.pets[link_house]} has a drink of {self.drinks[link_house]}."
        )
        clues.append(
            f"House {drink_direct_house} has a drink of {self.drinks[drink_direct_house]}."
        )

        self.rng.shuffle(clues)
        self.clue_sequence = clues

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()
        upper = text.upper()

        if upper == "CLUE":
            remaining = self.MAX_STEPS - self.step_count
            if self.clue_index < len(self.clue_sequence):
                clue = self.clue_sequence[self.clue_index]
                self.clue_index += 1
                obs = f"Clue {self.clue_index}/{len(self.clue_sequence)}: {clue} ({remaining} turns left.)"
            else:
                obs = (
                    "No clues remain -- you already have enough information to deduce the "
                    f"full solution. ({remaining} turns left.)"
                )
            truncated = self.step_count >= self.MAX_STEPS
            if truncated:
                self.done = True
            return obs, 0.0, False, truncated, {}

        if upper.startswith("SOLVE"):
            tokens = text.split()
            if len(tokens) != 7:
                obs = (
                    "Malformed SOLVE action. Use exactly: "
                    "SOLVE <pet1> <drink1> <pet2> <drink2> <pet3> <drink3>"
                )
                truncated = self.step_count >= self.MAX_STEPS
                if truncated:
                    self.done = True
                return obs, 0.0, False, truncated, {}

            guesses = tokens[1:]
            pet_correct = 0
            drink_correct = 0
            for h in (1, 2, 3):
                pet_guess = guesses[(h - 1) * 2].lower()
                drink_guess = guesses[(h - 1) * 2 + 1].lower()
                if pet_guess == self.pets[h].lower():
                    pet_correct += 1
                if drink_guess == self.drinks[h].lower():
                    drink_correct += 1

            reward = 0.5 * (pet_correct / 3.0) + 0.5 * (drink_correct / 3.0)
            self.done = True
            if pet_correct == 3 and drink_correct == 3:
                obs = "Correct! You matched every house's pet and drink."
            else:
                obs = (
                    f"Solution recorded: {pet_correct}/3 pets correct, "
                    f"{drink_correct}/3 drinks correct."
                )
            return obs, reward, True, False, {
                'pet_correct': pet_correct, 'drink_correct': drink_correct
            }

        obs = (
            "Unrecognized action. Send exactly 'CLUE' or "
            "'SOLVE <pet1> <drink1> <pet2> <drink2> <pet3> <drink3>'."
        )
        truncated = self.step_count >= self.MAX_STEPS
        if truncated:
            self.done = True
        return obs, 0.0, False, truncated, {}
