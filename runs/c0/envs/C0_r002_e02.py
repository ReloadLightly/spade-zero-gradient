import re
import random


class CaravanResupplyEnv:
    MAX_STEPS = 10
    NUM_LEGS = 5

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.leg_distances = [self.rng.randint(3, 8) for _ in range(self.NUM_LEGS)]
        self.prices = [self.rng.randint(2, 9) for _ in range(self.NUM_LEGS)]

        running_min = None
        prefix_min = []
        for p in self.prices:
            running_min = p if running_min is None else min(running_min, p)
            prefix_min.append(running_min)
        self.optimal_cost = sum(d * pm for d, pm in zip(self.leg_distances, prefix_min))

        slack = self.rng.randint(5, 15)
        self.budget = self.optimal_cost + slack
        self.gold = self.budget
        self.water = 0
        self.location = 0
        self.step_count = 0
        self.leg_reward_given = [False] * self.NUM_LEGS
        self.done = False

        legs_str = ", ".join(str(d) for d in self.leg_distances)
        obs = (
            f"You are the caravan master. Guide the caravan across {self.NUM_LEGS} desert legs "
            "to the oasis before gold or steps run out.\n"
            f"Starting gold: {self.gold}. Starting water: {self.water}.\n"
            f"Leg water requirements in order (leg 1..{self.NUM_LEGS}): [{legs_str}].\n"
            "Water prices differ at every waypoint and are revealed only when you arrive there; "
            "you may buy any amount of water at your current waypoint and unused water carries "
            "forward to later legs.\n"
            f"Current waypoint: 0 (start). Price of water here: {self.prices[0]} gold/unit.\n"
            "Actions: 'buy <amount>' to purchase water at the current waypoint, or 'advance' to "
            "travel the next leg (consumes water equal to that leg's requirement).\n"
            f"You have {self.MAX_STEPS} total actions. Reach the oasis with enough water on every "
            "leg while spending gold efficiently."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already ended.", 0.0, True, False, {}

        self.step_count += 1
        act = (action or "").strip().lower()
        reward = 0.0
        terminated = False

        buy_match = re.match(r"^buy\s+(\d+)$", act)

        if act == "advance":
            if self.location >= self.NUM_LEGS:
                obs = "You are already at the oasis."
            else:
                needed = self.leg_distances[self.location]
                if self.water < needed:
                    obs = (
                        f"Disaster: leg {self.location + 1} needs {needed} water but you carry "
                        f"only {self.water}. The caravan is stranded in the dunes."
                    )
                    terminated = True
                    self.done = True
                else:
                    self.water -= needed
                    self.location += 1
                    if not self.leg_reward_given[self.location - 1]:
                        reward += 0.1
                        self.leg_reward_given[self.location - 1] = True
                    if self.location == self.NUM_LEGS:
                        spent = self.budget - self.gold
                        eff = min(1.0, self.optimal_cost / spent) if spent > 0 else 1.0
                        reward += 0.5 * eff
                        terminated = True
                        self.done = True
                        obs = (
                            f"You reach the oasis! Total spent: {spent} gold (best possible with "
                            f"hindsight: {self.optimal_cost}). Gold remaining: {self.gold}."
                        )
                    else:
                        price = self.prices[self.location]
                        obs = (
                            f"Leg {self.location} complete, {self.water} water remaining. Now at "
                            f"waypoint {self.location}. Price here: {price} gold/unit. "
                            f"Gold: {self.gold}."
                        )
        elif buy_match:
            amount = int(buy_match.group(1))
            if self.location >= self.NUM_LEGS:
                obs = "There is no market at the oasis."
            elif amount <= 0:
                obs = "Buy amount must be a positive integer."
            else:
                price = self.prices[self.location]
                cost = amount * price
                if cost > self.gold:
                    affordable = self.gold // price
                    obs = (
                        f"Not enough gold: {amount} units would cost {cost} but you have "
                        f"{self.gold}. You could afford up to {affordable} units at this price."
                    )
                else:
                    self.gold -= cost
                    self.water += amount
                    obs = (
                        f"Bought {amount} water for {cost} gold. Now carrying {self.water} "
                        f"water, {self.gold} gold left."
                    )
        else:
            obs = "Malformed action. Use 'buy <amount>' or 'advance'."

        truncated = False
        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += " Step limit reached."

        return obs, reward, terminated, truncated, {}
