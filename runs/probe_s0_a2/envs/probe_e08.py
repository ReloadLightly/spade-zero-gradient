import random
import itertools


class FactoryChangeoverEnv:
    """Sequence factory jobs across three families to minimize hidden changeover cost."""

    FAMILIES = ('A', 'B', 'C')
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.steps = 0
        self.done = False

        family_pool = list(self.FAMILIES) * 2
        self.rng.shuffle(family_pool)
        self.job_ids = [f"J{i+1}" for i in range(6)]
        self.job_family = dict(zip(self.job_ids, family_pool))
        self.proc_time = {jid: self.rng.randint(3, 9) for jid in self.job_ids}

        self.setup = {}
        for f1 in self.FAMILIES:
            for f2 in self.FAMILIES:
                if f1 != f2:
                    self.setup[(f1, f2)] = self.rng.randint(2, 9)

        self.known = {}
        self._max_setup = max(self.setup.values())
        self._optimal_cost = self._compute_optimal()
        self._worst_cost = max(1, (len(self.job_ids) - 1) * self._max_setup)

        lines = [
            "FACTORY LINE SEQUENCING - maximize throughput by minimizing changeover cost.",
            f"{len(self.job_ids)} jobs must run on one machine, each belongs to a family (A, B, or C).",
            "Same-family changeovers are free (cost 0). Cross-family changeover costs are HIDDEN and ASYMMETRIC",
            "(cost from A to B need not equal cost from B to A) - you must probe them.",
            "",
            "Jobs (id: family, processing_time):",
        ]
        for jid in self.job_ids:
            lines.append(f"  {jid}: family {self.job_family[jid]}, proc_time {self.proc_time[jid]}")
        lines += [
            "",
            "Actions (exactly one per turn):",
            "  PROBE <F1> <F2>   - reveal the changeover cost from family F1 to family F2 (F1,F2 in A,B,C, F1 != F2)",
            "  SEQUENCE <id,id,id,id,id,id> - commit a final visiting order of ALL job ids (comma-separated, each exactly once); ends the episode",
            f"You have {self.MAX_STEPS} steps total. Reward: 0.2 for a valid final sequence, plus up to 0.8 for how close its total changeover cost is to the true minimum.",
        ]
        obs = "\n".join(lines)
        return obs, {}

    def _compute_optimal(self):
        best = None
        for perm in itertools.permutations(self.FAMILIES):
            cost = sum(self.setup[(perm[i], perm[i + 1])] for i in range(len(perm) - 1))
            if best is None or cost < best:
                best = cost
        return best

    def _known_table(self):
        if not self.known:
            return "No changeover costs probed yet."
        rows = [f"  {f1}->{f2}: {c}" for (f1, f2), c in sorted(self.known.items())]
        return "Known changeover costs:\n" + "\n".join(rows)

    def step(self, action):
        if self.done:
            return "Episode already ended. Reset to play again.", 0.0, True, False, {}

        self.steps += 1
        text = (action or "").strip()
        parts = text.split()
        reward = 0.0
        terminated = False
        info = {}

        if not parts:
            obs = "Empty action. Use 'PROBE <F1> <F2>' or 'SEQUENCE <id,id,...>'.\n" + self._known_table()
        elif parts[0].upper() == "PROBE":
            if len(parts) != 3:
                obs = "Malformed PROBE. Use exactly: PROBE <F1> <F2>\n" + self._known_table()
            else:
                f1, f2 = parts[1].upper(), parts[2].upper()
                if f1 not in self.FAMILIES or f2 not in self.FAMILIES:
                    obs = f"Unknown family letter. Families are {self.FAMILIES}.\n" + self._known_table()
                elif f1 == f2:
                    obs = "Same-family changeover is always 0 - no need to probe that.\n" + self._known_table()
                else:
                    cost = self.setup[(f1, f2)]
                    self.known[(f1, f2)] = cost
                    obs = f"Changeover cost {f1}->{f2} = {cost}.\n" + self._known_table()
        elif parts[0].upper() == "SEQUENCE":
            if len(parts) < 2:
                obs = "Malformed SEQUENCE. Use: SEQUENCE <id,id,id,id,id,id>\n" + self._known_table()
            else:
                raw = " ".join(parts[1:])
                ids = [x.strip().upper() for x in raw.replace(",", " ").split() if x.strip()]
                if sorted(ids) != sorted(self.job_ids):
                    obs = (
                        f"Invalid sequence: must list all {len(self.job_ids)} job ids exactly once "
                        f"({', '.join(self.job_ids)}).\n" + self._known_table()
                    )
                else:
                    fams = [self.job_family[j] for j in ids]
                    achieved = sum(
                        self.setup.get((fams[i], fams[i + 1]), 0)
                        for i in range(len(fams) - 1)
                        if fams[i] != fams[i + 1]
                    )
                    performance = max(0.0, 1.0 - (achieved - self._optimal_cost) / self._worst_cost)
                    performance = min(1.0, performance)
                    reward = 0.2 + 0.8 * performance
                    terminated = True
                    self.done = True
                    info = {"achieved_cost": achieved, "optimal_cost": self._optimal_cost}
                    obs = (
                        f"Sequence accepted: {','.join(ids)}\n"
                        f"Total changeover cost = {achieved} (true minimum possible = {self._optimal_cost}).\n"
                        f"Episode complete. Reward this episode: {reward:.3f}"
                    )
        else:
            obs = "Unrecognized action. Use 'PROBE <F1> <F2>' or 'SEQUENCE <id,id,...>'.\n" + self._known_table()

        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += f"\nStep limit ({self.MAX_STEPS}) reached without a committed sequence. Episode over."

        return obs, reward, terminated, truncated, info
