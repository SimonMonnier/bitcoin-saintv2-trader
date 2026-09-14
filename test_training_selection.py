import unittest
import json
import training as t
import numpy as np


class SelectionTests(unittest.TestCase):
    def test_legacy_curriculum_changes_actual_action_likelihood(self):
        p = np.array([1/3, 1/3, 1/3])
        behavior = t.rollout_action_probabilities(p, .87, 0., 'both')
        np.testing.assert_allclose(behavior, [.478333333333, .478333333333, .043333333333])
        self.assertGreater(abs(np.log(behavior[2])-np.log(p[2])), 2.)

    def test_corrected_sampling_preserves_model_probabilities(self):
        p = np.array([.12, .31, .57])
        for side in ('both', 'long', 'short'):
            np.testing.assert_allclose(t.rollout_action_probabilities(p, 0., 0., side), p)

    def test_hold_probability_includes_replaced_actions(self):
        p = np.array([.2, .45, .35])
        np.testing.assert_allclose(t.rollout_action_probabilities(p, 0., .3, 'both'), [0., .45, .55])

    def test_flat_policy_does_not_open_every_trade(self):
        values = np.full(1000, np.float32(1 / 3), dtype=np.float64)
        threshold = t.selective_threshold(values, .025)
        self.assertFalse(np.any(values >= threshold))
        restored = json.loads(json.dumps({'threshold': threshold}))['threshold']
        self.assertFalse(np.any(values >= restored))

    def test_weak_but_varying_policy_keeps_budget(self):
        values = np.linspace(.333, .334, 1000)
        threshold = t.selective_threshold(values, .025)
        self.assertEqual(int((values >= threshold).sum()), 25)

    def test_tied_tail_does_not_exceed_budget(self):
        values = np.r_[np.full(900, .33), np.full(100, .34)]
        self.assertLessEqual((values >= t.selective_threshold(values, .025)).mean(), .025)

    def test_invalid_samples_fail_explicitly(self):
        for values in ([], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                t.selective_threshold(values, .025)


if __name__ == '__main__':
    unittest.main()
