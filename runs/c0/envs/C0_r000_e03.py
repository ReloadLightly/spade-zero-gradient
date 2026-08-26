import random
import re


class EinsteinHousesEnv:
    COLORS = ['Red', 'Blue', 'Green']
    PETS = ['Dog', 'Cat', 'Bird']
    DRINKS = ['Tea', 'Coffee', 'Water']
    HOUSES = [1, 2, 3]
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        color_perm = self.rng.sample(self.COLORS, 3)
        pet_perm = self.rng.sample(self.PETS, 3)
        drink_perm = self.rng.sample(self.DRINKS, 3)
        self.solution = {
            h: {
                'color': color_perm[h - 1],
                'pet': pet_perm[h - 1],
                'drink': drink_perm[h - 1],
            }
            for h in self.HOUSES
        }

        direct_houses = self.rng.sample(self.HOUSES, 2)
        color_link_houses = self.rng.sample(self.HOUSES, 2)
        pet_link_houses = self.rng.sample(self.HOUSES, 2)

        clues = []
        for h in direct_houses:
            clues.append(f"House {h} has color {self.solution[h]['color']}.")
        for h in color_link_houses:
            clues.append(
                f"The house with color {self.solution[h]['color']} "
                f"has pet {self.solution[h]['pet']}."
            )
        for h in pet_link_houses:
            clues.append(
                f"The house with pet {self.solution[h]['pet']} "
                f"drinks {self.solution[h]['drink']}."
            )
        self.rng.shuffle(clues)
        self.clue_pool = clues
        self.next_clue_idx = 0

        self.achieved = {'color': False, 'pet': False, 'drink': False}
        self.reward_given = 0.0
        self.step_count = 0
        self.terminated = False

        revealed = []
        for _ in range(2):
            revealed.append(self.clue_pool[self.next_clue_idx])
            self.next_clue_idx += 1

        obs = (
            "EINSTEIN HOUSES. There are 3 houses (1, 2, 3). Each house has exactly "
            "one Color (Red, Blue, Green), one Pet (Dog, Cat, Bird), and one Drink "
            "(Tea, Coffee, Water); no value repeats across houses within a category.\n"
            "GOAL: determine the exact Color, Pet, and Drink of every house.\n"
            "ACTIONS:\n"
            "  ASK                -- reveal one more clue (if any remain)\n"
            "  SOLVE 1:<Color>,<Pet>,<Drink> 2:<Color>,<Pet>,<Drink> 3:<Color>,<Pet>,<Drink>\n"
            "                      -- submit your full grid; you may resubmit after ASKing more.\n"
            f"You have {self.MAX_STEPS} steps total (ASK and SOLVE both count).\n"
            "Correct attribute-blocks (all 3 houses right for Color, or for Pet, or for "
            "Drink) are locked in and rewarded once each; wrong blocks get a correctness "
            "count, not the answer.\n"
            "Known clues so far:\n- " + "\n- ".join(revealed)
        )
        return obs, {'clues_revealed': 2, 'clues_total': len(self.clue_pool)}

    def _format_status(self):
        parts = []
        for attr in ('color', 'pet', 'drink'):
            state = "LOCKED (correct)" if self.achieved[attr] else "not yet solved"
            parts.append(f"{attr}: {state}")
        return "; ".join(parts)

    def step(self, action):
        if self.terminated:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        action = (action or "").strip()
        reward = 0.0
        terminated = False

        if action.upper() == "ASK":
            if self.next_clue_idx < len(self.clue_pool):
                clue = self.clue_pool[self.next_clue_idx]
                self.next_clue_idx += 1
                remaining = len(self.clue_pool) - self.next_clue_idx
                obs = f"New clue: {clue}\nClues remaining in pool: {remaining}."
            else:
                obs = "No more clues available. Use SOLVE with what you know."
        elif action.upper().startswith("SOLVE"):
            entries = re.findall(
                r'(\d)\s*:\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)', action
            )
            houses_seen = {e[0] for e in entries}
            valid_format = (
                len(entries) == 3
                and houses_seen == {'1', '2', '3'}
            )
            guess = {}
            if valid_format:
                for h_str, c, p, d in entries:
                    h = int(h_str)
                    c, p, d = c.capitalize(), p.capitalize(), d.capitalize()
                    if c not in self.COLORS or p not in self.PETS or d not in self.DRINKS:
                        valid_format = False
                        break
                    guess[h] = {'color': c, 'pet': p, 'drink': d}

            if not valid_format:
                obs = (
                    "Malformed SOLVE. Format: SOLVE 1:<Color>,<Pet>,<Drink> "
                    "2:<Color>,<Pet>,<Drink> 3:<Color>,<Pet>,<Drink> using each house "
                    "number exactly once and valid Color/Pet/Drink names."
                )
            else:
                counts = {}
                for attr in ('color', 'pet', 'drink'):
                    counts[attr] = sum(
                        1 for h in self.HOUSES if guess[h][attr] == self.solution[h][attr]
                    )

                newly_achieved = []
                for attr in ('color', 'pet', 'drink'):
                    if not self.achieved[attr] and counts[attr] == 3:
                        self.achieved[attr] = True
                        newly_achieved.append(attr)

                if newly_achieved:
                    if all(self.achieved.values()):
                        reward = 1.0 - self.reward_given
                        terminated = True
                    else:
                        reward = len(newly_achieved) * (1.0 / 3.0)
                    self.reward_given += reward

                obs = (
                    f"Colors correct: {counts['color']}/3. "
                    f"Pets correct: {counts['pet']}/3. "
                    f"Drinks correct: {counts['drink']}/3.\n"
                    f"Status -> {self._format_status()}"
                )
                if terminated:
                    obs += "\nAll attribute-blocks solved. Puzzle complete."
        else:
            obs = "Unknown action. Use ASK or SOLVE 1:C,P,D 2:C,P,D 3:C,P,D."

        self.terminated = terminated
        truncated = (not terminated) and (self.step_count >= self.MAX_STEPS)
        info = {
            'clues_remaining': len(self.clue_pool) - self.next_clue_idx,
            'reward_given': self.reward_given,
        }
        return obs, reward, terminated, truncated, info
