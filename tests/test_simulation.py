import unittest

from splitmind.config import WorldConfig, make_config
from splitmind.environment.room_world import RoomWorld, Scenario
from splitmind.modules.memory import MemoryRecord
from splitmind.simulation import run_episode


def stale_memory_scenario():
    """Key now in room_f; memory confidently holds an old room_a record; observer reports room_f."""
    scenario = Scenario(agent_room="room_c", key_room="room_f", exit_room="room_c",
                        beacon_overrides={0: ("room_f", 0.95)}, initial_version=1)
    memory = MemoryRecord(value="room_a", timestamp=-5, world_version=0, confidence=0.99, confirmations=5)
    return scenario, memory


def controlled_cfg(strategy, latency=2):
    return make_config({"world": {"p_move": 0.0, "beacon_noise": 0.0},
                        "network": {"latency": latency},
                        "strategy": {"name": strategy}})


class TestEnvironment(unittest.TestCase):
    def test_deterministic_world(self):
        a, b = RoomWorld(WorldConfig(), seed=3), RoomWorld(WorldConfig(), seed=3)
        self.assertEqual((a.agent_room, a.key_room, a.exit_room), (b.agent_room, b.key_room, b.exit_room))
        self.assertEqual(a.schedule.key_moves, b.schedule.key_moves)

    def test_key_does_not_move_once_held(self):
        sc = Scenario(agent_room="room_b", key_room="room_b", exit_room="room_a",
                      key_moves={0: "room_e", 1: "room_f"})
        w = RoomWorld(WorldConfig(), seed=0, scenario=sc)
        w.step(("collect",))
        self.assertTrue(w.holding_key)
        self.assertEqual(w.version, 0)
        w.step(("move", "room_a"))
        w.step(("exit",))
        self.assertTrue(w.success)

    def test_key_move_increments_version(self):
        sc = Scenario(agent_room="room_a", key_room="room_c", exit_room="room_a", key_moves={1: "room_e"})
        w = RoomWorld(WorldConfig(), seed=0, scenario=sc)
        w.step(("wait",))
        w.step(("wait",))
        self.assertEqual((w.key_room, w.version, w.move_times), ("room_e", 1, [2]))

    def test_zero_latency_task_completion(self):
        cfg = make_config({"network": {"latency": 0}})
        for seed in range(20):
            summary, _ = run_episode(cfg, seed)
            self.assertEqual(summary["success"], 1, f"seed {seed}")


class TestDeterminism(unittest.TestCase):
    def test_seeded_runs_are_identical(self):
        cfg = make_config({"network": {"latency": 4}, "strategy": {"name": "freshness_conflict"}})
        s1, st1 = run_episode(cfg, seed=11, record_steps=True)
        s2, st2 = run_episode(cfg, seed=11, record_steps=True)
        self.assertEqual(st1, st2)
        self.assertEqual({k: v for k, v in s1.items() if v == v}, {k: v for k, v in s2.items() if v == v})

    def test_strategies_face_same_world(self):
        first = {}
        for name in ("fifo", "confidence", "freshness", "freshness_conflict"):
            cfg = make_config({"network": {"latency": 2}, "strategy": {"name": name}})
            _, steps = run_episode(cfg, seed=5, record_steps=True)
            first[name] = [(r["t"], r["true_key"]) for r in steps if r["seek"]][:3]
        self.assertEqual(len({tuple(v) for v in first.values()}), 1)


class TestStaleMemoryConflict(unittest.TestCase):
    def test_conflict_is_logged(self):
        scenario, memory = stale_memory_scenario()
        _, steps = run_episode(controlled_cfg("confidence"), seed=0, scenario=scenario,
                               memory_init=memory, record_steps=True)
        contested = [r for r in steps if r.get("contested")]
        self.assertTrue(contested)
        self.assertTrue(any(r["conflict"] and r["gt_label"] == "HAS_CONFLICT" for r in steps))
        self.assertEqual(contested[0]["workspace_belief"], "room_a")   # confidence-only keeps memory

    def test_newer_observation_overrides_older_memory(self):
        """With freshness-aware arbitration the fresh observation must win and spread."""
        for strategy in ("freshness", "freshness_conflict"):
            for seed in range(10):
                scenario, memory = stale_memory_scenario()
                summary, steps = run_episode(controlled_cfg(strategy), seed=seed, scenario=scenario,
                                             memory_init=memory, record_steps=True)
                # with jitter the stale recall can arrive after memory's updated recall,
                # in which case the workspace never sees the two side by side
                for r in steps:
                    if r.get("contested"):
                        self.assertEqual(r["workspace_belief"], "room_f", (strategy, seed))
                self.assertTrue(any(r["memory_belief"] == "room_f" for r in steps))
                self.assertTrue(any(r.get("all_correct") for r in steps), (strategy, seed))
                self.assertEqual(summary["success"], 1)


if __name__ == "__main__":
    unittest.main()
