import unittest

from splitmind.metrics.stats import mean_ci, paired_diff_ci, ratio_bootstrap, wilson_ci


class TestStats(unittest.TestCase):
    def test_wilson_all_successes_is_not_degenerate(self):
        p, lo, hi = wilson_ci(50, 50)
        self.assertEqual(p, 1.0)
        self.assertLess(lo, 0.95)
        self.assertAlmostEqual(hi, 1.0)

    def test_mean_ci_ignores_nan(self):
        m, h, n = mean_ci([1.0, 2.0, float("nan"), 3.0])
        self.assertEqual((m, n), (2.0, 3))
        self.assertGreater(h, 0)

    def test_ratio_bootstrap_point(self):
        pt, lo, hi = ratio_bootstrap([1, 2, 3], [2, 4, 6])
        self.assertAlmostEqual(pt, 0.5)
        self.assertLessEqual(lo, pt)
        self.assertGreaterEqual(hi, pt)

    def test_paired_diff(self):
        d, lo, hi = paired_diff_ci([1, 2, 3], [1, 2, 3])
        self.assertEqual((d, lo, hi), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
