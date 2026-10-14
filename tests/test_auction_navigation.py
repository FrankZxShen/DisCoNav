"""Settlement-to-navigation tests with no Habitat or model service dependency."""

from types import SimpleNamespace
import unittest

from src.auction_navigation import (
    consume_completed_goals, execution_goals, frontier_coordinates,
    install_settled_bundles, select_frontier_proposal,
)
from src.bundle_execution import BundleExecution
from src.target_auction import PriorityAuctionSystem


class AuctionNavigationTests(unittest.TestCase):
    def setUp(self):
        self.frontiers = {
            'frontier_12': '<centroid: (30, 40), number: 5>',
            'frontier_3': '<centroid: (10, 20), number: 8>',
        }
        self.execution = BundleExecution()

    def test_mapping_follows_numeric_labels_not_dictionary_order(self):
        self.assertEqual(frontier_coordinates(self.frontiers),
                         {'A': (10, 20), 'B': (30, 40)})
        with self.assertRaises(ValueError):
            frontier_coordinates({'frontier_0': (1, 2), 'frontier_1': (1, 2)})

    def test_entire_settled_bundle_reaches_local_planner_in_order(self):
        auction = PriorityAuctionSystem([1, 2], ['A', 'B'],
                                        {1: {'A': .7, 'B': .3},
                                         2: {'A': .2, 'B': .1}})
        auction.auction_process()
        result = auction.get_settled_result()
        self.assertEqual(result.bundles, {1: ('A', 'B'), 2: ()})
        self.assertAlmostEqual(result.payments[1], .24)
        self.assertAlmostEqual(result.net_utilities[1], .56)
        install_settled_bundles(self.execution, result, self.frontiers, 7)
        positions = {1: (1, 2), 2: (3, 4)}
        proposed = [[90, 90], [30, 40]]
        self.assertEqual(execution_goals(self.execution, proposed, positions, 50),
                         [[10, 20], [3, 4]])
        robots = [SimpleNamespace(navigation_goal_reached=True,
                                  navigation_goal_xy=(10, 20)),
                  SimpleNamespace(navigation_goal_reached=False)]
        consume_completed_goals(self.execution, robots)
        self.assertFalse(robots[0].navigation_goal_reached)
        self.assertEqual(execution_goals(self.execution, proposed, positions, 51),
                         [[30, 40], [3, 4]])
        # The old signal cannot consume the next waypoint a second time.
        consume_completed_goals(self.execution, robots)
        self.assertTrue(self.execution.has_pending())
        robots[0].navigation_goal_xy = (30, 40)
        robots[0].navigation_goal_reached = True
        consume_completed_goals(self.execution, robots)
        self.assertFalse(self.execution.has_pending())
        self.assertEqual(execution_goals(self.execution, proposed, positions, 52),
                         [[1, 2], [3, 4]])
        events = self.execution.drain_events()
        self.assertEqual([e['target'] for e in events if e['event'] == 'completed'],
                         ['A', 'B'])

    def test_coordinate_snapshot_survives_frontier_relabeling(self):
        result = SimpleNamespace(settled=True, bundles={1: ('A', 'B')})
        install_settled_bundles(self.execution, result, self.frontiers, 1)
        self.frontiers['frontier_3'] = '<centroid: (99, 99), number: 2>'
        self.assertEqual(self.execution.current(1), (10, 20))

    def test_unsettled_or_pending_batch_cannot_be_installed(self):
        with self.assertRaises(ValueError):
            install_settled_bundles(self.execution, None, self.frontiers, 1)
        with self.assertRaises(ValueError):
            install_settled_bundles(self.execution, SimpleNamespace(settled=False),
                                    self.frontiers, 1)
        result = SimpleNamespace(settled=True, bundles={1: ('A', 'B')})
        install_settled_bundles(self.execution, result, self.frontiers, 1)
        with self.assertRaises(RuntimeError):
            install_settled_bundles(self.execution, result, self.frontiers, 2)

    def test_recovery_completion_does_not_consume_pending_waypoint(self):
        self.execution.install({1: ('A', 'B')}, frontier_coordinates(self.frontiers), 1)
        self.execution.pause(1, 75)
        robot = SimpleNamespace(navigation_goal_reached=True,
                                navigation_goal_xy=(90, 90))
        consume_completed_goals(self.execution, [robot])
        self.assertEqual(execution_goals(self.execution, [[90, 90]], {1: (1, 2)}, 74),
                         [[90, 90]])
        # Signals describe the preceding action. Consume while still paused,
        # then resume the committed waypoint for the new action at step 75.
        robot.navigation_goal_reached = True
        consume_completed_goals(self.execution, [robot])
        self.assertEqual(execution_goals(self.execution, [[90, 90]], {1: (1, 2)}, 75),
                         [[10, 20]])
        robot.navigation_goal_reached = True
        robot.navigation_goal_xy = (10, 20)
        consume_completed_goals(self.execution, [robot])
        self.assertEqual(self.execution.current(1), (30, 40))

    def test_history_preemption_releases_only_that_robot(self):
        self.execution.install({1: ('A',), 2: ('B',)},
                               frontier_coordinates(self.frontiers), 1)
        self.execution.cancel(1, 'history_selected')
        self.assertEqual(execution_goals(self.execution, [[80, 80], [90, 90]],
                                         {1: (1, 2), 2: (3, 4)}, 75),
                         [[80, 80], [30, 40]])
        self.assertTrue(self.execution.has_pending())
        self.execution.clear('object_goal_found')
        self.assertFalse(self.execution.has_pending())
        self.assertEqual(execution_goals(self.execution, [[5, 6], [5, 6]],
                                         {1: (1, 2), 2: (3, 4)}, 76),
                         [[5, 6], [5, 6]])

    def test_unmanaged_invalid_goal_reaches_the_launcher_fallback(self):
        self.execution.install({1: ('A',)}, frontier_coordinates(self.frontiers), 1)
        proposed = [None, None]
        self.assertEqual(execution_goals(self.execution, proposed,
                                         {1: (1, 2), 2: (3, 4)}, 50),
                         [[10, 20], None])
        self.assertEqual(proposed, [None, None])


class FrontierProposalTests(unittest.TestCase):
    def setUp(self):
        self.frontiers = {
            'frontier_0': '<centroid: (10, 20), number: 8>',
            'frontier_1': '<centroid: (30, 40), number: 7>',
            'frontier_2': '<centroid: (50, 60), number: 6>',
            'frontier_3': '<centroid: (70, 80), number: 5>',
        }

    def test_single_frontier_is_available_to_all_preliminary_proposals(self):
        full = {'frontier_0': self.frontiers['frontier_0']}
        remaining = dict(full)
        for _ in range(3):
            self.assertEqual(select_frontier_proposal(full, remaining, [.1, .9, 0, 0]),
                             ('A', (10, 20)))
        self.assertEqual(remaining, {})
        self.assertEqual(len(full), 1)

    def test_three_robots_with_two_frontiers_exhaust_then_reuse_full_pool(self):
        full = {key: self.frontiers[key] for key in ('frontier_0', 'frontier_1')}
        remaining = dict(full)
        choices = [select_frontier_proposal(full, remaining, scores)
                   for scores in ([.7, .3, 0, 0], [.8, .2, 0, 0], [.1, .9, 0, 0])]
        self.assertEqual(choices, [('A', (10, 20)), ('B', (30, 40)), ('B', (30, 40))])
        self.assertEqual(remaining, {})

    def test_filtering_keeps_original_score_indices_and_letter_labels(self):
        remaining = {key: self.frontiers[key] for key in ('frontier_1', 'frontier_3')}
        self.assertEqual(select_frontier_proposal(self.frontiers, remaining, [.99, .1, .9, .8]),
                         ('D', (70, 80)))
        self.assertEqual(list(remaining), ['frontier_1'])
        self.assertEqual(select_frontier_proposal(self.frontiers, remaining, [.99, .1, .9, .8]),
                         ('B', (30, 40)))

    def test_four_frontiers_follow_scores_and_preserve_the_full_pool(self):
        remaining = dict(reversed(list(self.frontiers.items())))
        choices = [select_frontier_proposal(self.frontiers, remaining, [.1, .2, .4, .3])[0]
                   for _ in range(4)]
        self.assertEqual(choices, ['C', 'D', 'B', 'A'])
        self.assertEqual(len(self.frontiers), 4)

    def test_invalid_scores_or_pool_do_not_remove_candidates(self):
        remaining = dict(self.frontiers)
        for scores in ([.1], [.1, .2, float('nan'), .4]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                select_frontier_proposal(self.frontiers, remaining, scores)
            self.assertEqual(remaining, self.frontiers)
        with self.assertRaises(ValueError):
            select_frontier_proposal({}, {}, [])


if __name__ == '__main__':
    unittest.main()
