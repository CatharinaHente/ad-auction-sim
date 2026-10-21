"""Fair platform: Jain-constrained fairness wrapper on top of standard GSP."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import Bounds, minimize

from sim.platform import Platform


class PlatformFair(Platform):

    def __init__(self, bidders, delta=0.1, high_stakes_ad="STEM"):
        super().__init__(bidders)
        self.delta = delta
        self.high_stakes_ad = high_stakes_ad
        self._group_total: dict[str, int] = {}
        self._group_hs:    dict[str, int] = {}
        self._target_shares: dict[str, float] = {}
        self.history: list[dict] = []

    @staticmethod
    def jain(values: list[float]) -> float:
        n, s, sq = len(values), sum(values), sum(v**2 for v in values)
        return s**2 / (n * sq) if sq > 0 else 1.0

    def _current_shares(self) -> dict[str, float]:
        return {
            g: self._group_hs.get(g, 0) / max(self._group_total.get(g, 1), 1)
            for g in self._group_total
        }

    def _solve_targets(self) -> dict[str, float]:
        groups = sorted(self._group_total)
        n = len(groups)
        if n < 2:
            return {}
        current = self._current_shares()
        shares = [current[g] for g in groups]
        if self.jain(shares) >= 1 - self.delta:
            return {g: current[g] for g in groups}
        def objective(r):
            return sum(max(0.0, r[i] - shares[i]) for i in range(n))
        def jain_con(r):
            return self.jain(list(r)) - (1 - self.delta)
        lb = np.array(shares)
        result = minimize(
            objective, lb, method="SLSQP",
            bounds=Bounds(lb=lb, ub=np.ones(n)),
            constraints={"type": "ineq", "fun": jain_con},
            options={"ftol": 1e-9},
        )
        if result.success:
            return {groups[i]: float(np.clip(result.x[i], lb[i], 1.0)) for i in range(n)}
        return {g: float(max(shares)) for g in groups}

    def _override_prob(self, group: str) -> float:
        target = self._target_shares.get(group, 0.0)
        return float(target)

    def assign(self, adspots, method="gsp", valuation_fn=None):
        results = []
        for spot in adspots:
            group = spot.tags[0] if spot.tags else "unknown"

            # Step 1: run standard GSP via parent (keeps Qs=1, everything unchanged)
            [res] = super().assign([spot], method=method, valuation_fn=valuation_fn)

            # Step 2: override if GSP didn't show high-stakes ad
            winner = res["winners"][0]
            gsp_shows_hs = winner is not None and winner.name == self.high_stakes_ad

            override_applied = False
            if not gsp_shows_hs and random.random() < self._override_prob(group):
                hs = next((b for b in self.bidders if b.name == self.high_stakes_ad), None)
                if hs:
                    res = {**res, "winners": [hs], "override": True}
                    override_applied = True

            shown_hs = gsp_shows_hs or override_applied

            # Step 3: update stats and re-solve
            self._group_total[group] = self._group_total.get(group, 0) + 1
            self._group_hs[group]    = self._group_hs.get(group, 0) + int(shown_hs)
            self._target_shares = self._solve_targets()

            shares = self._current_shares()
            self.history.append({"jain": self.jain(list(shares.values())), "shares": dict(shares)})
            results.append(res)
        return results

    def fairness_report(self):
        shares = self._current_shares()
        return {
            "impression_shares_hs": shares,
            "jain_index": self.jain(list(shares.values())),
            "target_shares": dict(self._target_shares),
        }