import re
import random


class MysteryGuestHatEnv:
    COLORS = ['Red', 'Blue', 'Green', 'Yellow']
    FAMILIES = {'Red': 'Warm', 'Yellow': 'Warm', 'Blue': 'Cool', 'Green': 'Cool'}
    MAX_STEPS = 10
    SHRINK_UNIT = 0.2
    ASK_CAP = 0.6
    GUESS_REWARD = 0.4

    ASK_RE = re.compile(r'^ASK\s+(\d+)$', re.IGNORECASE)
    GUESS_RE = re.compile(r'^GUESS\s+([A-Za-z]+)$', re.IGNORECASE)

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.target_color = self.rng.choice(self.COLORS)
        others = [c for c in self.COLORS if c != self.target_color]
        self.rng.shuffle(others)
        visible = others[:]
        visible += [self.rng.choice(self.COLORS) for _ in range(2)]
        self.rng.shuffle(visible)
        self.visible_colors = visible

        self.clue_texts = []
        self.clue_elim = []
        for i, vc in enumerate(visible):
            if vc != self.target_color:
                elim = {vc}
                text = (f'Player {i + 1} (hat: {vc}) says: "I can state with certainty '
                        f'the Mystery Guest\'s hat is NOT {vc}."')
            else:
                fam = self.FAMILIES[vc]
                other_fam = 'Cool' if fam == 'Warm' else 'Warm'
                elim = {c for c in self.COLORS if self.FAMILIES[c] == other_fam}
                text = (f'Player {i + 1} (hat: {vc}) says: "I can state with certainty the '
                        f'Mystery Guest\'s hat belongs to the {fam} family (Warm = Red/Yellow, '
                        f'Cool = Blue/Green), not the {other_fam} family."')
            self.clue_texts.append(text)
            self.clue_elim.append(elim)

        self.candidates = set(self.COLORS)
        self.asked = set()
        self.ask_reward_total = 0.0
        self.steps = 0

        lines = ["A hat has vanished from the Mystery Guest and you must name its color."]
        lines.append(f"Palette of possible colors: {self.COLORS}.")
        lines.append("Five witnesses stand before you, each wearing a visible hat:")
        for i, vc in enumerate(visible):
            lines.append(f"  Player {i + 1}: {vc} hat")
        lines.append("Each witness, if asked, will make one truthful statement ruling out "
                      "some possibilities for the Mystery Guest's hat color.")
        lines.append("Actions: 'ASK <player number 1-5>' to hear a witness's statement, or "
                      "'GUESS <color>' to name the Mystery Guest's hat color.")
        lines.append("A GUESS ends the investigation immediately, right or wrong. "
                      f"You have at most {self.MAX_STEPS} actions total.")
        obs = "\n".join(lines)
        return obs, {}

    def step(self, action):
        self.steps += 1
        text = (action or "").strip()
        reward = 0.0
        terminated = False

        ask_m = self.ASK_RE.match(text)
        guess_m = self.GUESS_RE.match(text)

        if ask_m:
            pid = int(ask_m.group(1))
            if pid < 1 or pid > 5:
                obs = "Invalid player number. Choose a witness numbered 1 to 5."
            elif (pid - 1) in self.asked:
                obs = f"Player {pid} has already spoken. Ask someone else or make your GUESS."
            else:
                idx = pid - 1
                self.asked.add(idx)
                before = len(self.candidates)
                self.candidates -= self.clue_elim[idx]
                shrink = before - len(self.candidates)
                if shrink > 0:
                    give = min(self.SHRINK_UNIT * shrink, self.ASK_CAP - self.ask_reward_total)
                    give = max(give, 0.0)
                    reward = round(give, 4)
                    self.ask_reward_total += reward
                obs = (self.clue_texts[idx] +
                       f"\n(Remaining possible colors: {sorted(self.candidates)})")
        elif guess_m:
            guess_raw = guess_m.group(1)
            found = next((c for c in self.COLORS if c.lower() == guess_raw.lower()), None)
            if found is None:
                obs = f"'{guess_raw}' is not a color in the palette {self.COLORS}."
            else:
                terminated = True
                if found == self.target_color:
                    reward = self.GUESS_REWARD
                    obs = f"Correct! The Mystery Guest's hat is {self.target_color}. Case closed."
                else:
                    reward = 0.0
                    obs = (f"Incorrect. You guessed {found}, but the Mystery Guest's hat was "
                            f"{self.target_color}. Case closed.")
        else:
            obs = "Malformed action. Use 'ASK <player number 1-5>' or 'GUESS <color>'."

        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            obs += "\n\nStep limit reached. The case is closed unsolved."

        return obs, reward, terminated, truncated, {}
