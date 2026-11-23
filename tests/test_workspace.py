import unittest

from splitmind.config import StrategyConfig
from splitmind.workspace import (ConfidenceStrategy, FIFOStrategy, FreshnessConflictStrategy,
                                 FreshnessStrategy, GlobalWorkspace)
from tests.helpers import msg

NOW = 10


def spec_conflict():
    """The conflict example from the study design: fresh observer vs confident old memory."""
    obs = msg("observer", "room_c", created_at=10, confidence=0.95, sent_at=10)
    mem = msg("memory", "room_a", created_at=5, confidence=0.99, sent_at=10)
    return obs, mem


class TestArbitration(unittest.TestCase):
    def test_strategies_differ_on_controlled_conflict(self):
        obs, mem = spec_conflict()
        cands = sorted([obs, mem], key=lambda m: m.source)
        self.assertEqual(ConfidenceStrategy().select(cands, [obs, mem], None, NOW).value, "room_a")
        self.assertEqual(FreshnessStrategy().select(cands, [obs, mem], None, NOW).value, "room_c")
        self.assertEqual(FreshnessConflictStrategy().select(cands, [obs, mem], None, NOW).value, "room_c")
        # FIFO depends only on arrival order
        self.assertEqual(FIFOStrategy().select(cands, [obs, mem], None, NOW).value, "room_a")
        self.assertEqual(FIFOStrategy().select(cands, [mem, obs], None, NOW).value, "room_c")

    def test_fifo_keeps_incumbent_without_arrivals(self):
        obs, _ = spec_conflict()
        self.assertIs(FIFOStrategy().select([obs], [], obs, NOW), obs)

    def test_proposed_resists_weak_fresh_dissent(self):
        # weak fresh report vs strong recent memory: freshness follows the fresh report,
        # the proposed score keeps the memory (0.99 * e^-0.1 = 0.90 > 0.6; no penalty since 0.6 < tau)
        weak = msg("observer", "room_c", created_at=10, confidence=0.6)
        mem = msg("memory", "room_a", created_at=9, confidence=0.99)
        cands = [mem, weak]
        self.assertEqual(FreshnessStrategy().select(cands, [], None, NOW).value, "room_c")
        self.assertEqual(FreshnessConflictStrategy().select(cands, [], None, NOW).value, "room_a")

    def test_conflict_penalty(self):
        obs, mem = spec_conflict()
        s = FreshnessConflictStrategy(StrategyConfig(lam=0.0, delta=0.5, tau=0.9))
        self.assertEqual(s.conflict(mem, [obs, mem]), 1)
        self.assertEqual(s.conflict(obs, [obs, mem]), 0)        # memory is older -> no penalty
        self.assertAlmostEqual(s.score(mem, [obs, mem], NOW), 0.99 * 0.5)
        s_lowtau = FreshnessConflictStrategy(StrategyConfig(tau=0.99))
        self.assertEqual(s_lowtau.conflict(mem, [obs, mem]), 0)  # 0.95 is below tau

    def test_relevance_and_additive_form(self):
        a = msg("observer", "room_c", created_at=10, confidence=0.9, relevance=0.2)
        b = msg("memory", "room_a", created_at=8, confidence=0.9)
        mult = FreshnessConflictStrategy(StrategyConfig(lam=0.1, delta=0.0))
        self.assertEqual(mult.select([a, b], [], None, NOW).value, "room_a")
        add = FreshnessConflictStrategy(StrategyConfig(form="additive", alpha=0.0, beta=1.0,
                                                       gamma=0.1, delta=0.0))
        self.assertEqual(add.select([a, b], [], None, NOW).value, "room_c")

    def test_workspace_end_to_end_tick(self):
        obs, mem = spec_conflict()
        ws = GlobalWorkspace(FreshnessConflictStrategy())
        ws.receive([mem, obs])
        self.assertEqual(ws.arbitrate(NOW).value, "room_c")
        b = ws.broadcast(NOW)
        self.assertEqual(b.value, "room_c")
        self.assertEqual({s["source"] for s in b.content["slots"]}, {"memory", "observer"})


if __name__ == "__main__":
    unittest.main()
