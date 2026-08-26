import random
import re


class FamilyTreeDeductionEnv:
    NAME_POOL = [
        "Ada", "Beno", "Cass", "Dov", "Ester", "Farin", "Gael", "Hira",
        "Iker", "Jora", "Kael", "Lira", "Moro", "Nils", "Orla", "Pell",
    ]
    ROLES = ["GF", "GM", "P1", "P2", "P3", "M1", "M3", "C1", "C2", "C3"]
    RELATION_WORDS = {
        "parent", "child", "grandparent", "grandchild",
        "sibling", "aunt_or_uncle", "niece_or_nephew", "cousin",
    }

    def __init__(self):
        self.max_steps = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        names = self.rng.sample(self.NAME_POOL, 10)
        role = dict(zip(self.ROLES, names))
        self.all_people = list(names)

        self.parents = {}
        for child_role, par_roles in (
            ("P1", ("GF", "GM")), ("P2", ("GF", "GM")), ("P3", ("GF", "GM")),
            ("C1", ("P1", "M1")), ("C2", ("P3", "M3")), ("C3", ("P3", "M3")),
        ):
            self.parents[role[child_role]] = [role[par_roles[0]], role[par_roles[1]]]

        self.spouse = {}
        for x, y in (("GF", "GM"), ("P1", "M1"), ("P3", "M3")):
            self.spouse[role[x]] = role[y]
            self.spouse[role[y]] = role[x]

        self.children = {p: [] for p in self.all_people}
        for c, plist in self.parents.items():
            for p in plist:
                self.children[p].append(c)

        candidates = [
            (a, b) for a in self.all_people for b in self.all_people
            if a != b and self._relation(a, b) is not None
        ]
        self.query_a, self.query_b = self.rng.choice(candidates)
        self.correct_relation = self._relation(self.query_a, self.query_b)

        self.relevant = self._neighbors2(self.query_a) | self._neighbors2(self.query_b)
        self.reward_per = 0.4 / len(self.relevant)

        self.asked = set()
        self.steps = 0
        self.done = False

        obs = (
            f"A family has {len(self.all_people)} members: {', '.join(self.all_people)}.\n"
            f"Determine how {self.query_a} is related to {self.query_b} "
            f"(the answer must describe what {self.query_a} IS to {self.query_b}).\n"
            f"Valid relation words: {', '.join(sorted(self.RELATION_WORDS))}.\n"
            "Actions (exactly one per turn):\n"
            "  ASK <name>     - reveal that person's recorded parents, spouse, and children\n"
            "  ANSWER <word>  - submit your final relation guess (ends the episode)\n"
            f"You have {self.max_steps} total actions."
        )
        return obs, {"query_a": self.query_a, "query_b": self.query_b}

    def _ancestors_map(self, x):
        result = {}
        for p in self.parents.get(x, []):
            result.setdefault(p, 1)
            for gp in self.parents.get(p, []):
                result.setdefault(gp, 2)
        return result

    def _relation(self, a, b):
        if a == b:
            return None
        anc_a = self._ancestors_map(a)
        anc_b = self._ancestors_map(b)
        if b in anc_a:
            return "child" if anc_a[b] == 1 else "grandchild"
        if a in anc_b:
            return "parent" if anc_b[a] == 1 else "grandparent"
        common = set(anc_a) & set(anc_b)
        if not common:
            return None
        best = min(common, key=lambda c: anc_a[c] + anc_b[c])
        da, db = anc_a[best], anc_b[best]
        if da == 1 and db == 1:
            return "sibling"
        if da == 1 and db == 2:
            return "aunt_or_uncle"
        if da == 2 and db == 1:
            return "niece_or_nephew"
        if da == 2 and db == 2:
            return "cousin"
        return None

    def _neighbors2(self, x):
        s = {x}
        s |= set(self.parents.get(x, []))
        s |= set(self.children.get(x, []))
        for p in self.parents.get(x, []):
            s |= set(self.children.get(p, []))
            s |= set(self.parents.get(p, []))
        for c in self.children.get(x, []):
            s |= set(self.children.get(c, []))
        return s

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        remaining = self.max_steps - self.steps
        text = action.strip() if isinstance(action, str) else ""

        ask_match = re.match(r'^ASK\s+(.+)$', text, re.IGNORECASE)
        answer_match = re.match(r'^ANSWER\s+(.+)$', text, re.IGNORECASE)

        if ask_match:
            name_raw = ask_match.group(1).strip()
            match_name = next(
                (p for p in self.all_people if p.lower() == name_raw.lower()), None
            )
            if match_name is None:
                obs = (f"No family member named '{name_raw}'. "
                        f"Valid names: {', '.join(self.all_people)}. ({remaining} actions left)")
                return obs, 0.0, False, self.steps >= self.max_steps, {}

            plist = self.parents.get(match_name, [])
            sp = self.spouse.get(match_name)
            clist = self.children.get(match_name, [])
            obs = (
                f"Record for {match_name} — "
                f"parents: {', '.join(plist) if plist else 'none recorded'}; "
                f"spouse: {sp if sp else 'none recorded'}; "
                f"children: {', '.join(clist) if clist else 'none recorded'}. "
                f"({remaining} actions left)"
            )
            reward = 0.0
            if match_name not in self.asked and match_name in self.relevant:
                reward = self.reward_per
            self.asked.add(match_name)
            return obs, reward, False, self.steps >= self.max_steps, {}

        if answer_match:
            guess = answer_match.group(1).strip().lower()
            if guess not in self.RELATION_WORDS:
                obs = (f"'{guess}' is not a valid relation word. "
                        f"Valid words: {', '.join(sorted(self.RELATION_WORDS))}. ({remaining} actions left)")
                return obs, 0.0, False, self.steps >= self.max_steps, {}

            self.done = True
            if guess == self.correct_relation:
                obs = f"Correct! {self.query_a} is the {guess} of {self.query_b}."
                return obs, 0.6, True, False, {}
            obs = "Incorrect final answer. Episode over."
            return obs, 0.0, True, False, {}

        obs = (f"Malformed action. Use 'ASK <name>' or 'ANSWER <relation_word>'. "
                f"({remaining} actions left)")
        return obs, 0.0, False, self.steps >= self.max_steps, {}
