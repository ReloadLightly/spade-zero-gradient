import re
import random
import itertools


class WitnessTimelineEnv:
    """Reconstruct a 5-event murder timeline from witness before/after fragments."""

    EVENT_POOL = [
        "the study lamp was switched off",
        "the garden gate was left unlocked",
        "the housekeeper polished the silver",
        "a scream was heard from the east wing",
        "the family portrait was found tilted",
        "the wine cellar door creaked open",
        "the telephone line went dead",
        "the guest's dog began barking",
    ]

    MAX_STEPS = 10
    ASK_BUDGET = 0.6
    ORDER_BUDGET = 0.4

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.codes = ["E1", "E2", "E3", "E4", "E5"]
        chosen = self.rng.sample(self.EVENT_POOL, 5)
        self.descriptions = dict(zip(self.codes, chosen))

        self.true_order = list(self.codes)
        self.rng.shuffle(self.true_order)

        edges = [(self.true_order[i], self.true_order[i + 1]) for i in range(4)]
        consecutive = {frozenset(e) for e in edges}
        remaining = [p for p in itertools.combinations(self.codes, 2)
                     if frozenset(p) not in consecutive]
        for a, b in self.rng.sample(remaining, 2):
            if self.true_order.index(a) < self.true_order.index(b):
                edges.append((a, b))
            else:
                edges.append((b, a))

        witness_nums = list(range(1, 7))
        self.rng.shuffle(witness_nums)
        self.witness_statements = {witness_nums[i]: edges[i] for i in range(6)}

        self.known_edges = set()
        self.asked = set()
        self.steps = 0
        self.prev_determined = 0
        self.done = False

        lines = ["A manor murder must be sequenced from fragmentary accounts.",
                 "Five events occurred, each exactly once, in one hidden order.",
                 "Six witnesses each hold one 'X happened before Y' fact.",
                 "", "EVENTS:"]
        for c in self.codes:
            lines.append(f"  {c}: {self.descriptions[c]}")
        lines += [
            "",
            "ACTIONS (exactly one per turn):",
            "  ASK <n>      -- question witness n (n = 1 to 6) for their account.",
            "  ORDER <c1>,<c2>,<c3>,<c4>,<c5>  -- deliver your final chronological",
            "                  accusation, earliest first, using all 5 codes once.",
            "                  This immediately ends the investigation, so gather",
            "                  evidence before you submit it.",
            "",
            f"You have {self.MAX_STEPS} steps total; each ASK or ORDER uses one.",
        ]
        return "\n".join(lines), {}

    def _reachable(self):
        reach = {c: set() for c in self.codes}
        for u, v in self.known_edges:
            reach[u].add(v)
        for _ in range(len(self.codes)):
            for u in self.codes:
                for v in list(reach[u]):
                    reach[u] |= reach.get(v, set())
        return reach

    def _determined_count(self):
        reach = self._reachable()
        count = 0
        for a, b in itertools.combinations(self.codes, 2):
            if b in reach[a] or a in reach[b]:
                count += 1
        return count

    def step(self, action):
        if self.done:
            return "The investigation has already concluded.", 0.0, True, False, {}

        text = (action or "").strip()
        self.steps += 1
        reward = 0.0
        terminated = False

        ask_match = re.match(r"^ASK\s+(\d+)\s*$", text, re.IGNORECASE)
        order_match = re.match(r"^ORDER\s+(.+)$", text, re.IGNORECASE)

        if ask_match:
            n = int(ask_match.group(1))
            if n not in self.witness_statements:
                obs = f"There is no witness numbered {n}. Valid witnesses are 1-6."
            else:
                u, v = self.witness_statements[n]
                revisit = n in self.asked
                self.asked.add(n)
                self.known_edges.add((u, v))
                new_determined = self._determined_count()
                gained = new_determined - self.prev_determined
                if gained > 0:
                    reward = self.ASK_BUDGET * (gained / 10.0)
                self.prev_determined = new_determined
                lead = "You revisit witness" if revisit else "Witness"
                obs = (f"{lead} {n}: \"{self.descriptions[u]}\" ({u}) happened "
                       f"before \"{self.descriptions[v]}\" ({v}).")
                obs += (" This narrows the timeline." if gained > 0
                        else " This adds no new information beyond what you know.")
        elif order_match:
            tokens = [t for t in re.split(r"[,\s]+", order_match.group(1).strip().upper()) if t]
            if len(tokens) != 5 or set(tokens) != set(self.codes):
                obs = ("Invalid accusation: list all five codes E1-E5 exactly once, "
                       "separated by commas.")
            else:
                correct = sum(1 for i in range(5) if tokens[i] == self.true_order[i])
                reward = self.ORDER_BUDGET * (correct / 5.0)
                terminated = True
                self.done = True
                obs = (f"You accuse the timeline: {','.join(tokens)}. "
                       f"Correct positions: {correct}/5. "
                       f"The true order was {','.join(self.true_order)}.")
        else:
            obs = ("Unrecognized action. Use 'ASK <n>' (n=1-6) or "
                   "'ORDER <c1>,<c2>,<c3>,<c4>,<c5>'.")

        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += " Time has run out; the investigation is truncated."

        return obs, reward, terminated, truncated, {"steps": self.steps}
