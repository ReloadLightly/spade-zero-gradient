import re
import random


class NightBridgeWeakPlankEnv:
    SECTION_NAMES = {1: "near", 2: "mid", 3: "far"}
    SECTION_ALIASES = {
        "near": 1, "n": 1, "1": 1,
        "mid": 2, "middle": 2, "m": 2, "2": 2,
        "far": 3, "f": 3, "3": 3,
    }

    def __init__(self):
        self.rng = None
        self.true_plank = None
        self.true_section = None
        self.tested_sections = {}
        self.milestone_given = False
        self.narrow_reward_given = False
        self.narrowed_correct = False
        self.steps = 0
        self.max_steps = 10
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.true_plank = self.rng.randint(1, 9)
        self.true_section = (self.true_plank - 1) // 3 + 1
        self.tested_sections = {}
        self.milestone_given = False
        self.narrow_reward_given = False
        self.narrowed_correct = False
        self.steps = 0
        self.done = False
        obs = (
            "NIGHT CROSSING — a rope bridge has 9 numbered planks (1-9) grouped into three fixed "
            "sections: NEAR (1-3), MID (4-6), FAR (7-9). One plank somewhere on the bridge is rotten; "
            "its exact number is never told to you directly. Your torch allows 10 actions total "
            "(mistakes count too) before it dies.\n"
            "ACTIONS (send exactly one per turn):\n"
            "  listen <near|mid|far>  - press weight on that section; hear CREAK if the rotten plank "
            "is in it, QUIET otherwise.\n"
            "  narrow <near|mid|far>  - declare which section you believe hides the rotten plank.\n"
            "  cross <1-9>            - lead the party across on that exact plank; this ends the crossing.\n"
            "Goal: end on the true rotten plank for full credit. Efficient listening and a correct "
            "section declaration also earn credit on their own, even if your final footing is wrong. "
            "Malformed actions are corrected with no reward but still use up a turn."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "The crossing has already ended; start a new episode to try again.", 0.0, True, False, {}

        self.steps += 1
        remaining_after = max(self.max_steps - self.steps, 0)

        m = re.match(r'^\s*(listen|narrow|cross)\s+([A-Za-z0-9]+)\s*$', action.strip(), re.IGNORECASE)
        terminated = False
        if not m:
            obs = (
                "Malformed action. Use 'listen <near|mid|far>', 'narrow <near|mid|far>', or "
                f"'cross <1-9>'. ({remaining_after} actions left.)"
            )
            reward = 0.0
        else:
            verb = m.group(1).lower()
            arg = m.group(2).lower()

            if verb == "cross":
                if not arg.isdigit() or not (1 <= int(arg) <= 9):
                    obs = f"Plank must be a number from 1 to 9. ({remaining_after} actions left.)"
                    reward = 0.0
                else:
                    plank = int(arg)
                    plank_section = (plank - 1) // 3 + 1
                    terminated = True
                    self.done = True
                    if plank == self.true_plank:
                        reward = 0.5
                        obs = (
                            f"CRACK — but you step past it just in time. Plank {plank} was indeed "
                            f"rotten; the party crosses safely."
                        )
                    elif self.narrowed_correct and plank_section == self.true_section:
                        reward = 0.15
                        obs = (
                            f"The plank splinters — plank {plank} was sound, but the rotten one "
                            f"({self.true_plank}) was hiding elsewhere in the same section. Some "
                            f"scramble back to safety."
                        )
                    else:
                        reward = 0.0
                        obs = (
                            f"The bridge gives way. Plank {plank} was sound, and the rotten plank "
                            f"({self.true_plank}) was nowhere near your guess."
                        )
            elif arg not in self.SECTION_ALIASES:
                obs = f"Unknown section '{arg}'. Use near, mid, or far. ({remaining_after} actions left.)"
                reward = 0.0
            else:
                section = self.SECTION_ALIASES[arg]
                if verb == "listen":
                    result = "CREAK" if section == self.true_section else "QUIET"
                    self.tested_sections[section] = result
                    reward = 0.0
                    if len(self.tested_sections) >= 2 and not self.milestone_given:
                        reward = 0.2
                        self.milestone_given = True
                    obs = (
                        f"You press down on the {self.SECTION_NAMES[section]} section: {result}. "
                        f"({remaining_after} actions left.)"
                    )
                else:
                    correct = (section == self.true_section)
                    self.narrowed_correct = correct
                    reward = 0.0
                    if correct and not self.narrow_reward_given:
                        reward = 0.3
                        self.narrow_reward_given = True
                    obs = (
                        f"You commit to believing the rotten plank is in the "
                        f"{self.SECTION_NAMES[section]} section. ({remaining_after} actions left.)"
                    )

        truncated = False
        if not terminated and self.steps >= self.max_steps:
            truncated = True
            self.done = True
            obs = obs + " The torch gutters out — the crossing attempt is over."

        return obs, reward, terminated, truncated, {}
