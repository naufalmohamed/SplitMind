import unittest

from splitmind.config import MemoryConfig, SelfMonitorConfig
from splitmind.metrics.coherence import has_conflict, pairwise_coherence
from splitmind.metrics.metacognition import evaluate, ground_truth_label, macro_f1
from splitmind.metrics.task import recovery_times, spells, wrong_action_cause
from splitmind.modules.memory import Memory
from splitmind.modules.planner import Planner
from splitmind.modules.self_monitor import SelfMonitor
from splitmind.workspace import ConfidenceStrategy, GlobalWorkspace
from tests.helpers import msg

ROOMS = ["room_a", "room_b", "room_c", "room_d", "room_e", "room_f"]


class TestMemory(unittest.TestCase):
    def test_world_version_follows_evidence(self):
        mem = Memory(MemoryConfig())
        mem.receive([msg("observer", "room_a", 0, 0.8, version=0)])
        mem.receive([msg("observer", "room_c", 4, 0.8, version=1)])
        self.assertEqual(mem.belief, "room_c")
        self.assertEqual(mem.belief_version, 1)
        recall = mem.rehearse(9)
        self.assertEqual(recall.created_at, 4)      # keeps evidence time, not send time
        self.assertEqual(recall.sent_at, 9)
        self.assertEqual(recall.world_version, 1)

    def test_late_older_evidence_is_ignored(self):
        mem = Memory(MemoryConfig())
        mem.receive([msg("observer", "room_c", 5, 0.8, version=1)])
        mem.receive([msg("observer", "room_a", 2, 0.99, version=0)])   # arrives late
        self.assertEqual(mem.belief, "room_c")
        self.assertEqual(mem.belief_version, 1)

    def test_consolidation_raises_confidence(self):
        mem = Memory(MemoryConfig(consolidate=True, max_confidence=0.99))
        for t in range(3):
            mem.receive([msg("observer", "room_a", t, 0.8)])
        self.assertAlmostEqual(mem.record.confidence, 0.99)   # 1 - 0.2**3 = 0.992, capped
        mem.receive([msg("observer", "room_c", 5, 0.8)])
        self.assertAlmostEqual(mem.record.confidence, 0.8)    # new value resets

    def test_no_consolidation(self):
        mem = Memory(MemoryConfig(consolidate=False))
        for t in range(3):
            mem.receive([msg("observer", "room_a", t, 0.8)])
        self.assertAlmostEqual(mem.record.confidence, 0.8)


class TestWorkspaceSlots(unittest.TestCase):
    def test_slot_keeps_newest_evidence_per_source(self):
        ws = GlobalWorkspace(ConfidenceStrategy())
        ws.receive([msg("observer", "room_c", 6, 0.8)])
        ws.receive([msg("observer", "room_a", 3, 0.99)])   # older, out of order
        self.assertEqual(ws.slots["observer"].value, "room_c")


class TestCoherence(unittest.TestCase):
    def test_pairwise(self):
        coh, n_conf, n_distinct = pairwise_coherence(
            {"observer": "room_c", "memory": "room_a", "workspace": "room_a", "planner": "room_a"})
        self.assertAlmostEqual(coh, 3 / 6)
        self.assertEqual(n_conf, 3)
        self.assertEqual(n_distinct, 2)

    def test_none_is_not_comparable(self):
        coh, n_conf, _ = pairwise_coherence({"observer": "room_c", "memory": None, "workspace": "room_c",
                                             "planner": None})
        self.assertEqual(coh, 1.0)
        self.assertEqual(n_conf, 0)
        coh, _, _ = pairwise_coherence({"observer": "room_c", "memory": None})
        self.assertNotEqual(coh, coh)   # NaN: fewer than two comparable modules

    def test_conflict_detection(self):
        self.assertTrue(has_conflict({"a": "room_a", "b": "room_c"}))
        self.assertFalse(has_conflict({"a": "room_a", "b": "room_a", "c": None}))


class TestGroundTruthLabels(unittest.TestCase):
    def test_labels(self):
        base = {"observer": "room_c", "memory": "room_c", "workspace": "room_c", "planner": "room_c"}
        self.assertEqual(ground_truth_label(base, "room_c"), "KNOWS")
        self.assertEqual(ground_truth_label(base, "room_a"), "MAY_BE_STALE")
        self.assertEqual(ground_truth_label({**base, "memory": "room_a"}, "room_c"), "HAS_CONFLICT")
        self.assertEqual(ground_truth_label({**base, "workspace": None}, "room_c"), "DOES_NOT_KNOW")


class TestSelfMonitor(unittest.TestCase):
    def _snapshot(self, value, conf, created_at, slots, sent_at=0):
        from splitmind.messages import Message
        return Message(source="workspace",
                       content={"key_location": value,
                                "winner": {"source": "observer", "confidence": conf, "created_at": created_at},
                                "slots": slots},
                       created_at=created_at, sent_at=sent_at, confidence=conf, world_version=0,
                       message_id=f"ws:{sent_at}")

    def test_states(self):
        sm = SelfMonitor(SelfMonitorConfig(hazard=0.1, stale_threshold=0.5))
        self.assertEqual(sm.assess(0)["state"], "DOES_NOT_KNOW")
        fresh = [{"source": "observer", "value": "room_c", "confidence": 0.99, "created_at": 5}]
        sm.receive([self._snapshot("room_c", 0.99, 5, fresh, sent_at=5)])
        out = sm.assess(5)
        self.assertEqual(out["state"], "KNOWS")
        self.assertAlmostEqual(out["confidence"], 0.99)
        self.assertEqual(sm.assess(20)["state"], "MAY_BE_STALE")   # 0.99 * 0.9**15 < 0.5
        split = fresh + [{"source": "memory", "value": "room_a", "confidence": 0.9, "created_at": 1}]
        sm.receive([self._snapshot("room_c", 0.99, 5, split, sent_at=6)])
        out = sm.assess(6)
        self.assertEqual(out["state"], "HAS_CONFLICT")
        self.assertAlmostEqual(out["confidence"], 0.9)

    def test_naive_always_certain(self):
        sm = SelfMonitor(SelfMonitorConfig(), variant="naive")
        slots = [{"source": "observer", "value": "room_c", "confidence": 0.8, "created_at": 0},
                 {"source": "memory", "value": "room_a", "confidence": 0.99, "created_at": 0}]
        sm.receive([self._snapshot("room_c", 0.8, 0, slots)])
        self.assertEqual(sm.assess(30), {"state": "KNOWS", "confidence": 0.8})


class TestPlanner(unittest.TestCase):
    def test_rules(self):
        p = Planner(ROOMS, exit_room="room_a")
        self.assertEqual(p.act("room_b", holding_key=True), ("move", "room_a"))
        self.assertEqual(p.act("room_a", holding_key=True), ("exit",))
        self.assertEqual(p.act("room_b", holding_key=False), ("move", "room_c"))   # explore
        p.receive([msg("workspace", "room_e", 0, 0.8)])
        self.assertEqual(p.act("room_c", holding_key=False), ("move", "room_d"))
        self.assertEqual(p.act("room_e", holding_key=False), ("collect",))


class TestMetrics(unittest.TestCase):
    def test_wrong_action_causes(self):
        opt = ("move", "room_d")
        self.assertIsNone(wrong_action_cause(("move", "room_d"), opt, "room_f", 0, "room_f", 1))
        self.assertEqual(wrong_action_cause(("move", "room_b"), opt, "room_a", 0, "room_f", 1), "stale")
        self.assertEqual(wrong_action_cause(("move", "room_b"), opt, "room_a", 1, "room_f", 1), "noise")
        self.assertEqual(wrong_action_cause(("move", "room_b"), opt, None, None, "room_f", 1), "unknown")

    def test_spells_and_recovery(self):
        self.assertEqual(spells([True, True, False, True, False, False, True]), [2, 1, 1])
        ok = {t: t in (3, 4, 9) for t in range(12)}
        self.assertEqual(recovery_times([1, 6, 10], ok, 12), [2, 3, None])

    def test_macro_f1_and_rates(self):
        y = ["KNOWS", "KNOWS", "HAS_CONFLICT", "MAY_BE_STALE"]
        p = ["KNOWS", "HAS_CONFLICT", "KNOWS", "MAY_BE_STALE"]
        # KNOWS f1 = 0.5, HAS_CONFLICT f1 = 0, MAY_BE_STALE f1 = 1
        self.assertAlmostEqual(macro_f1(y, p), 0.5)
        r = evaluate(y, p, [0.9, 0.6, 0.85, 0.7])
        self.assertAlmostEqual(r["accuracy"], 0.5)
        self.assertAlmostEqual(r["false_certainty_rate"], 0.5)
        self.assertAlmostEqual(r["critical_failure_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
