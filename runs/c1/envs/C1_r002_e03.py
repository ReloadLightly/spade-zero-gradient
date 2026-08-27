import re


class AuthenticGemBalanceEnv:
    LABELS = ['A', 'B', 'C', 'D', 'E', 'F']
    COMPARE_RE = re.compile(r'^\s*compare\s+([A-Fa-f])\s+([A-Fa-f])\s*$')
    GUESS_RE = re.compile(r'^\s*guess\s+([A-Fa-f])\s*$')

    def __init__(self):
        self.rng = None
        self.weights = {}
        self.genuine_label = None
        self.step_count = 0
        self.max_steps = 10
        self.proven_forgery = set()
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)

        heavy_size = self.rng.randint(2, 3)
        light_size = 5 - heavy_size
        base = self.rng.randint(40, 60)
        heavy_w = base + self.rng.randint(5, 15)
        light_w = base - self.rng.randint(5, 15)

        weights_list = [heavy_w] * heavy_size + [light_w] * light_size + [base]
        self.rng.shuffle(weights_list)
        self.weights = dict(zip(self.LABELS, weights_list))
        self.genuine_label = next(l for l, w in self.weights.items() if w == base)

        self.step_count = 0
        self.proven_forgery = set()
        self.done = False

        obs = (
            "APPRAISER'S SCALE: Six gems, labeled A through F, sit before you. "
            "Exactly one is authentic; the other five are convincing forgeries. "
            "Forgeries are not all alike -- they belong to two hidden weight "
            "families (some heavier, some lighter than the truth), but every "
            "forgery has at least one twin of identical weight somewhere in "
            "the set. The authentic gem has no twin.\n"
            "Use a balance scale to compare two gems at a time. "
            "Action format: 'compare X Y' (e.g. 'compare A C') reports whether "
            "X is heavier, Y is heavier, or they balance. "
            "When ready, submit 'guess X' with your final answer -- this ends "
            "the episode immediately, right or wrong. "
            "You have at most 10 actions total, including your guess."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        text = action or ""
        m = self.COMPARE_RE.match(text)
        g = self.GUESS_RE.match(text)

        if m:
            x, y = m.group(1).upper(), m.group(2).upper()
            if x == y:
                obs = f"Malformed action: '{action}'. Pick two different gems, e.g. 'compare A B'."
            else:
                wx, wy = self.weights[x], self.weights[y]
                if wx == wy:
                    obs = f"Balanced: {x} and {y} weigh the same."
                    newly = set()
                    for label in (x, y):
                        if label != self.genuine_label and label not in self.proven_forgery:
                            newly.add(label)
                    self.proven_forgery |= newly
                    reward += 0.08 * len(newly)
                elif wx > wy:
                    obs = f"{x} is heavier than {y}."
                else:
                    obs = f"{y} is heavier than {x}."
        elif g:
            guess = g.group(1).upper()
            terminated = True
            self.done = True
            if guess == self.genuine_label:
                reward += 0.6
                obs = f"Correct! {guess} is the authentic gem. Episode complete."
            else:
                obs = f"Incorrect. {guess} was a forgery. The authentic gem was {self.genuine_label}. Episode complete."
        else:
            obs = f"Malformed action: '{action}'. Use 'compare X Y' or 'guess X' with letters A-F."

        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            self.done = True
            obs += " Step limit reached; episode over without a guess."

        info = {'step': self.step_count, 'proven_forgery_count': len(self.proven_forgery)}
        return obs, reward, terminated, truncated, info
