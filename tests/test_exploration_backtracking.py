import unittest

import numpy as np

from src.exploration_backtracking import (
    circular_heading_indices,
    directional_history_evidence,
    history_node_score,
    navigation_mode,
    should_explore_frontier,
    write_directional_evidence,
)


class ExplorationBacktrackingTests(unittest.TestCase):
    def test_paper_numerical_example(self):
        evidence = directional_history_evidence(0.89, 0.53, 2.0, 1.0)
        self.assertAlmostEqual(evidence, 2.31)

        state = np.zeros(360)
        write_directional_evidence(state, 180, evidence)
        self.assertEqual(np.count_nonzero(state), 78)
        self.assertAlmostEqual(history_node_score(state, 1), 180.18)

    def test_every_boundary_heading_has_exactly_78_unique_bins(self):
        for heading in (0, 1, 38, 39, 320, 321, 359):
            with self.subTest(heading=heading):
                indices = circular_heading_indices(heading)
                self.assertEqual(len(indices), 78)
                self.assertEqual(len(set(indices.tolist())), 78)

                state = np.zeros(360)
                write_directional_evidence(state, heading, 1.0)
                self.assertEqual(np.count_nonzero(state), 78)

    def test_wrapping_window_is_centered_on_zero(self):
        indices = set(circular_heading_indices(0).tolist())
        self.assertEqual(indices, set(range(321, 360)) | set(range(0, 39)))

    def test_existing_state_uses_same_window_as_new_state(self):
        new_state = np.zeros(360)
        existing_state = np.zeros(360)
        write_directional_evidence(new_state, 20, 0.7)
        write_directional_evidence(existing_state, 20, 0.7)
        np.testing.assert_array_equal(new_state, existing_state)

    def test_warmup_is_inclusive(self):
        self.assertTrue(should_explore_frontier(0.0, 0, beta=0.3, warmup_steps=125))
        self.assertTrue(should_explore_frontier(0.0, 125, beta=0.3, warmup_steps=125))
        self.assertFalse(should_explore_frontier(0.0, 126, beta=0.3, warmup_steps=125))

    def test_post_warmup_switch_compares_js_directly(self):
        self.assertFalse(should_explore_frontier(0.2999, 150, beta=0.3))
        self.assertTrue(should_explore_frontier(0.3, 150, beta=0.3))
        self.assertEqual(navigation_mode(0.2, 150, beta=0.3), "history")
        self.assertEqual(navigation_mode(0.8, 150, beta=0.3), "frontier")

    def test_history_score_uses_visit_count(self):
        state = np.ones(360)
        self.assertEqual(history_node_score(state, 4), 90.0)
        with self.assertRaises(ValueError):
            history_node_score(state, 0)


if __name__ == "__main__":
    unittest.main()
