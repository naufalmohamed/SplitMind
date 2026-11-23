import math
import unittest
from dataclasses import replace

from splitmind.config import MemoryConfig, SelfMonitorConfig, StrategyConfig, WorldConfig, make_config
from splitmind.environment.room_world import RoomWorld, Scenario
from splitmind.messages import Message
from splitmind.modules.memory import Memory
from splitmind.modules.observer import Observer
from splitmind.modules.self_monitor import SelfMonitor
from splitmind.phase2.complex_world import ComplexWorldConfig, RingWorld, ring_step, run_complex_episode
from splitmind.phase2.multi_source import ObserverSpec, RemoteObserver, run_multi_source_episode
from splitmind.phase2.runner import run_p2_grid, selection_stats, timing_stats
from splitmind.simulation import run_episode
from splitmind.workspace import FreshnessStrategy, GlobalWorkspace, ReliabilityAwareStrategy
from tests.helpers import msg

ROOMS = ["room_a", "room_b", "room_c", "room_d", "room_e", "room_f"]


def rel_cfg(**kw):
    base = dict(name="reliability", form="linear", w_r=1.0, w_c=0.0, w_a=0.0, w_g=0.0,
                reliabilities={"obs_a": 0.95, "obs_b": 0.6, "obs_c": 0.6})
    base.update(kw)
    return StrategyConfig(**base)


class TestObservationCorruption(unittest.TestCase):
    def test_sight_noise_zero_reports_truth(self):
        sc = Scenario(agent_room="room_c", key_room="room_c", exit_room="room_a")
        w = RoomWorld(WorldConfig(), seed=0, scenario=sc)
        out = Observer(make_config().observer).observe(w.observe())
        self.assertEqual(out[0].value, "room_c")

    def test_sight_errors_name_a_wrong_room(self):
        cfg = WorldConfig(sight_noise=1.0, p_move=0.0)
        w = RoomWorld(cfg, seed=4)
        self.assertEqual(len(w.schedule.sight_errors), cfg.max_steps + 1)
        self.assertTrue(all(r != w.key_room for r in w.schedule.sight_errors.values()))
        w.agent_room = w.key_room
        o = w.observe()
        self.assertTrue(o.key_visible)
        self.assertNotEqual(o.sight, w.key_room)

    def test_sight_noise_does_not_change_phase1_streams(self):
        a, b = RoomWorld(WorldConfig(), seed=9), RoomWorld(WorldConfig(sight_noise=0.5), seed=9)
        self.assertEqual(a.schedule.key_moves, b.schedule.key_moves)
        self.assertEqual(a.schedule.beacon_errors, b.schedule.beacon_errors)
        self.assertEqual((a.agent_room, a.key_room, a.exit_room), (b.agent_room, b.key_room, b.exit_room))


class TestIndependentWorldChanges(unittest.TestCase):
    def test_facts_change_independently(self):
        w = RingWorld(ComplexWorldConfig(p_key=0.0, p_exit=1.0, p_block=0.0), seed=1)
        key, block = w.key, w.block
        for _ in range(5):
            w.step(("wait",))
        self.assertEqual(w.version, {"key_location": 0, "exit_location": 5, "blocked_edge": 0})
        self.assertEqual((w.key, w.block), (key, block))

    def test_blocked_edge_stops_movement_and_routing_avoids_it(self):
        w = RingWorld(ComplexWorldConfig(p_key=0, p_exit=0, p_block=0), seed=2)
        w.agent, w.block = "room_a", "room_a|room_b"
        w.step(("move", "room_b"))
        self.assertEqual((w.agent, w.failed_moves), ("room_a", 1))
        rooms = w.rooms
        self.assertEqual(ring_step(rooms, "room_a", "room_c", None), "room_b")
        self.assertEqual(ring_step(rooms, "room_a", "room_c", "room_a|room_b"), "room_h")


class TestMultiSource(unittest.TestCase):
    def test_per_observer_reliability(self):
        rooms = ROOMS
        for r in (0.95, 0.6):
            o = RemoteObserver(ObserverSpec("obs_x", r, latency=0), 0, seed=3, rooms=rooms, horizon=4000)
            reports = [o.observe(t, "room_c", 0)[0].value for t in range(4000)]
            acc = sum(v == "room_c" for v in reports) / len(reports)
            self.assertAlmostEqual(acc, r, delta=0.03)

    def test_confidence_calibration(self):
        self.assertAlmostEqual(ObserverSpec("a", 0.6, 0, calibration=0.3).confidence, 0.9)
        self.assertAlmostEqual(ObserverSpec("a", 0.9, 0, calibration=0.3).confidence, 0.99)

    def test_source_specific_latency(self):
        obs = [ObserverSpec("obs_a", 1.0, latency=5), ObserverSpec("obs_c", 1.0, latency=0)]
        cfg = make_config({"world": {"p_move": 0.0, "sight_noise": 0.0}, "network": {"jitter_frac": 0.0},
                           "strategy": {"name": "freshness"}})
        _, steps = run_multi_source_episode(cfg, obs, seed=0, internal_latency=1, record_steps=True)
        first = next(r for r in steps if r.get("selected_message_source"))
        self.assertEqual(first["selected_message_source"], "obs_c")
        self.assertEqual(first["t"], 0)

    def test_multi_observer_conflict(self):
        a = Message("obs_a", {"key_location": "room_c", "origin": "obs_a"}, created_at=7, sent_at=7,
                    confidence=0.95, world_version=0, message_id="a")
        b = Message("obs_b", {"key_location": "room_b", "origin": "obs_b"}, created_at=10, sent_at=10,
                    confidence=0.6, world_version=0, message_id="b")
        c = Message("obs_c", {"key_location": "room_c", "origin": "obs_c"}, created_at=9, sent_at=9,
                    confidence=0.6, world_version=0, message_id="c")
        ws = GlobalWorkspace(FreshnessStrategy())
        ws.receive([a, b, c])
        self.assertEqual(ws.arbitrate(10).value, "room_b")
        ws.strategy = ReliabilityAwareStrategy(rel_cfg(w_g=1.0))
        self.assertEqual(ws.arbitrate(10).value, "room_c")


class TestReliabilityArbitration(unittest.TestCase):
    def setUp(self):
        self.a = Message("obs_a", {"key_location": "room_c"}, 7, 7, 0.95, 0, "a")
        self.b = Message("obs_b", {"key_location": "room_b"}, 10, 10, 0.6, 0, "b")
        self.c = Message("obs_c", {"key_location": "room_c"}, 9, 9, 0.6, 0, "c")
        self.mem = Message("memory", {"key_location": "room_b", "origin": "obs_a"}, 5, 10, 0.99, 0, "m")

    def test_agreement_excludes_memory_and_self(self):
        s = ReliabilityAwareStrategy(rel_cfg())
        cands = [self.a, self.b, self.c, self.mem]
        self.assertAlmostEqual(s.agreement(self.a, cands), 0.6 / (0.6 + 0.6))
        self.assertAlmostEqual(s.agreement(self.b, cands), 0.0)
        # memory credited with its origin's reliability; origin excluded from its agreement set
        self.assertAlmostEqual(s.reliability(self.mem), 0.95)
        self.assertAlmostEqual(s.agreement(self.mem, cands), 0.6 / 1.2)

    def test_linear_score(self):
        s = ReliabilityAwareStrategy(rel_cfg(w_r=1.0, w_c=0.5, w_a=0.1, w_g=2.0))
        cands = [self.a, self.b, self.c]
        expected = 0.95 + 0.5 * 0.95 - 0.1 * 3 + 2.0 * 0.5
        self.assertAlmostEqual(s.score(self.a, cands, 10), expected)

    def test_bayes_vote(self):
        s = ReliabilityAwareStrategy(rel_cfg(form="bayes", lam=0.0))
        cands = [self.a, self.b, self.c, self.mem]
        chosen = s.select(cands, [], None, 10)
        self.assertEqual(chosen.value, "room_c")       # 0.95 + 0.6 outweigh 0.6
        self.assertIs(chosen, self.c)                  # freshest message for the winning value
        weak = ReliabilityAwareStrategy(rel_cfg(form="bayes", lam=0.0, reliabilities={}))
        self.assertEqual(weak.select([self.a, self.b], [], None, 10).value, "room_b")   # tie -> fresher


class TestSelfMonitorPriors(unittest.TestCase):
    def _snap(self, value, created_at, sent_at, conf=0.8):
        return Message("workspace", {"key_location": value,
                                     "winner": {"source": "observer", "confidence": conf, "created_at": created_at},
                                     "slots": [{"source": "observer", "value": value, "confidence": conf,
                                                "created_at": created_at}]},
                       created_at, sent_at, conf, 0, f"ws:{sent_at}")

    def test_incorrect_prior_changes_assessment(self):
        low = SelfMonitor(replace(SelfMonitorConfig(), hazard=0.05))
        high = SelfMonitor(replace(SelfMonitorConfig(), hazard=0.3))
        for m in (low, high):
            m.receive([self._snap("room_c", 0, 0)])
        self.assertEqual(low.assess(3)["state"], "KNOWS")          # 0.8 * 0.95^3 = 0.69
        self.assertEqual(high.assess(3)["state"], "MAY_BE_STALE")  # 0.8 * 0.7^3 = 0.27

    def test_no_prior_monitor_estimates_hazard(self):
        m = SelfMonitor(SelfMonitorConfig(hazard=0.01), "inferred")
        self.assertAlmostEqual(m.hazard, 0.5)          # Laplace prior before any evidence
        values = ["room_a", "room_a", "room_b", "room_b", "room_b"]
        for t, v in enumerate(values):
            m.receive([self._snap(v, t, t)])
        self.assertAlmostEqual(m.hazard, (1 + 1) / (4 + 2))
        m.receive([self._snap("room_c", 1, 2)])        # older snapshot: ignored
        self.assertAlmostEqual(m.hazard, 2 / 6)


class TestAccumulatingMemory(unittest.TestCase):
    def test_single_contradiction_does_not_flip(self):
        mem = Memory(MemoryConfig(mode="accumulate", decay=0.1))
        for t in range(4):
            mem.receive([msg("observer", "room_a", t, 0.8)])
        mem.receive([msg("observer", "room_c", 4, 0.8)])
        self.assertEqual(mem.belief, "room_a")
        latest = Memory(MemoryConfig())
        for t in range(4):
            latest.receive([msg("observer", "room_a", t, 0.8)])
        latest.receive([msg("observer", "room_c", 4, 0.8)])
        self.assertEqual(latest.belief, "room_c")

    def test_origin_follows_stored_evidence(self):
        """Regression (validation finding F1): an ignored same-time report must not take over provenance."""
        mem = Memory(MemoryConfig())
        strong = Message("obs_a", {"key_location": "room_c", "origin": "obs_a"}, 5, 5, 0.95, 0, "a")
        weak = Message("obs_b", {"key_location": "room_c", "origin": "obs_b"}, 5, 5, 0.55, 0, "b")
        mem.receive([strong, weak])
        self.assertEqual((mem.record.confidence, mem.record.origin), (0.95, "obs_a"))
        newer = Message("obs_b", {"key_location": "room_c", "origin": "obs_b"}, 6, 6, 0.55, 0, "c")
        mem.receive([newer])                               # newer agreeing evidence: consolidated
        self.assertEqual(mem.record.origin, "obs_b")
        older = Message("obs_c", {"key_location": "room_d", "origin": "obs_c"}, 2, 2, 0.99, 0, "d")
        mem.receive([older])                               # older evidence: ignored
        self.assertEqual((mem.record.value, mem.record.origin), ("room_c", "obs_b"))

    def test_topic_aware_recall(self):
        mem = Memory(MemoryConfig(), topic="exit_location")
        mem.receive([Message("observer", {"exit_location": "room_d"}, 0, 0, 0.8, 0, "x", topic="exit_location")])
        r = mem.rehearse(1)
        self.assertEqual((r.topic, r.value), ("exit_location", "room_d"))


class TestPhase2Metrics(unittest.TestCase):
    def test_selection_and_timing_stats(self):
        rows = [
            {"t": 0, "selected_source": "", "selected_stale": False, "selected_fresh_wrong": False,
             "memory_slot_correct": False, "observer_slot_correct": False, "observer_slot_value": None,
             "memory_slot_value": None, "conflict": False, "wrong_cause": "unknown"},
            {"t": 1, "selected_source": "memory", "selected_stale": True, "selected_fresh_wrong": False,
             "memory_slot_correct": False, "observer_slot_correct": True, "observer_slot_value": "room_c",
             "memory_slot_value": "room_a", "conflict": True, "wrong_cause": ""},
            {"t": 2, "selected_source": "observer", "selected_stale": False, "selected_fresh_wrong": False,
             "memory_slot_correct": False, "observer_slot_correct": True, "observer_slot_value": "room_c",
             "memory_slot_value": "room_a", "conflict": True, "wrong_cause": ""},
            {"t": 3, "selected_source": "observer", "selected_stale": False, "selected_fresh_wrong": True,
             "memory_slot_correct": True, "observer_slot_correct": False, "observer_slot_value": "room_b",
             "memory_slot_value": "room_c", "conflict": True, "wrong_cause": "noise"},
        ]
        s = selection_stats(rows)
        self.assertEqual((s["sel_steps"], s["stale_sel"], s["fresh_wrong_sel"]), (3, 1, 1))
        self.assertAlmostEqual(s["observer_correct_selection"], 0.5)
        self.assertAlmostEqual(s["memory_correct_selection"], 0.0)
        t = timing_stats(rows)
        self.assertEqual((t["first_sustained_conflict"], t["first_wrong_any"], t["first_wrong_belief"]), (1, 0, 3))


class TestPhase2Determinism(unittest.TestCase):
    def test_multi_source_reproducible(self):
        obs = [ObserverSpec("obs_a", 0.9, 3, 2), ObserverSpec("obs_c", 0.6, 0)]
        cfg = make_config({"strategy": {"name": "reliability", "form": "bayes"}})
        a = run_multi_source_episode(cfg, obs, seed=5, record_steps=True)
        b = run_multi_source_episode(cfg, obs, seed=5, record_steps=True)
        self.assertEqual(a[1], b[1])

    def test_complex_reproducible(self):
        cfg = make_config({"network": {"latency": 2}})
        a = run_complex_episode(cfg, ComplexWorldConfig(), seed=7, record_steps=True)
        b = run_complex_episode(cfg, ComplexWorldConfig(), seed=7, record_steps=True)
        self.assertEqual(a[1], b[1])

    def test_runner_rows_reproducible(self):
        grid = {"network.latency": [2], "_s": [{"_strategy": "freshness", "strategy.name": "freshness"}]}
        a, _ = run_p2_grid({}, grid, seeds=3, progress=False)
        b, _ = run_p2_grid({}, grid, seeds=3, progress=False)
        self.assertTrue(a.equals(b))

    def test_detail_flag_does_not_change_outcome(self):
        cfg = make_config({"network": {"latency": 4}})
        s1, _ = run_episode(cfg, seed=3)
        s2, _ = run_episode(cfg, seed=3, record_steps=True, detail=True,
                            extra_monitors={"m": SelfMonitor(cfg.self_monitor, "inferred")})
        same = {k: v for k, v in s1.items() if not (isinstance(v, float) and math.isnan(v))}
        self.assertEqual(same, {k: s2[k] for k in same})


if __name__ == "__main__":
    unittest.main()
