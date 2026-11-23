import unittest

from splitmind.config import NetworkConfig
from splitmind.network.simulator import Network
from tests.helpers import msg


class TestMessage(unittest.TestCase):
    def test_age_is_measured_from_evidence_time(self):
        m = msg("memory", "room_a", created_at=3, sent_at=9, confidence=0.9)
        self.assertEqual(m.age(10), 7)
        self.assertEqual(m.age(3), 0)

    def test_transit_delay(self):
        m = msg("observer", "room_a", created_at=2, confidence=0.9).delivered(6)
        self.assertEqual(m.delivered_at, 6)
        self.assertEqual(m.transit_delay, 4)


class TestNetwork(unittest.TestCase):
    def test_delayed_delivery(self):
        net = Network(NetworkConfig(latency=3, jitter_frac=0.0), seed=0, horizon=20)
        net.send(msg("observer", "room_a", 0, 0.9), "observer", "workspace", now=0)
        for t in range(3):
            self.assertEqual(net.deliver("workspace", t), [], f"delivered too early at t={t}")
        out = net.deliver("workspace", 3)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].delivered_at, 3)
        self.assertEqual(net.deliver("workspace", 4), [])

    def test_zero_latency_same_tick(self):
        net = Network(NetworkConfig(latency=0), seed=0, horizon=5)
        net.send(msg("observer", "room_a", 1, 0.9), "observer", "memory", now=1)
        self.assertEqual(len(net.deliver("memory", 1)), 1)

    def test_same_tick_messages_keep_send_order(self):
        net = Network(NetworkConfig(latency=2, jitter_frac=0.0), seed=0, horizon=10)
        a = msg("observer", "room_a", 0, 0.9, mid="a")
        b = msg("memory", "room_b", 0, 0.9, mid="b")
        net.send(a, "observer", "workspace", now=0)
        net.send(b, "memory", "workspace", now=0)
        self.assertEqual([m.message_id for m in net.deliver("workspace", 2)], ["a", "b"])

    def test_delivery_sorted_by_delivery_time(self):
        net = Network(NetworkConfig(latency=1, jitter_frac=0.0,
                                    link_latency={"memory->workspace": 4}), seed=0, horizon=10)
        net.send(msg("memory", "room_a", 0, 0.9, mid="slow"), "memory", "workspace", now=0)
        net.send(msg("observer", "room_b", 1, 0.9, mid="fast"), "observer", "workspace", now=1)
        out = net.deliver("workspace", 10)
        self.assertEqual([m.message_id for m in out], ["fast", "slow"])

    def test_jitter_bounds_and_determinism(self):
        cfg = NetworkConfig(latency=4, jitter_frac=0.5)

        def delays(seed):
            net = Network(cfg, seed=seed, horizon=50)
            return [net.send(msg("observer", "room_a", t, 0.9), "observer", "workspace", t).transit_delay
                    for t in range(50)]

        d = delays(7)
        self.assertTrue(all(2 <= x <= 6 for x in d))
        self.assertGreater(len(set(d)), 1)
        self.assertEqual(d, delays(7))
        self.assertNotEqual(d, delays(8))


if __name__ == "__main__":
    unittest.main()
