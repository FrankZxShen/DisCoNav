"""Exercise the real discussion/settlement path with deterministic VLM outputs."""

import json
import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault('cv2', types.ModuleType('cv2'))

from src.GDP_agent import (
    Coordinator, RobotAgent, calculate_R3_weighted_scores,
    optimal_weighted_combination,
)


class FakeVLM:
    def COT3(self, **kwargs):
        if kwargs['return_string_probabilities'] is None:
            return None, json.dumps({
                'rebuttal': {'rebuttal_reason': 'comparison'},
                'required_adjustment': {'new_target': 'B', 'recommendation': 'consider B'},
            })
        return [0., 1., 0., 0.], 'B'


class CoordinatorTests(unittest.TestCase):
    def make_coordinator(self, **parameters):
        vlm = FakeVLM()
        robots = [RobotAgent(i, vlm) for i in (1, 2)]
        frontiers = {'frontier_0': '<centroid: (0, 0), number: 5>',
                     'frontier_1': '<centroid: (10, 0), number: 5>'}
        for robot in robots:
            robot.navigation_mode = 'frontier'
            robot.target_point = 'A'
            robot.target_object = 'chair'
            robot.DS = .8
            robot.JS = .8
            robot.origin_Decision_PR = [1., 0., 0., 0.] if robot.id == 1 else [.2, .8, 0., 0.]
            robot.frontier_coords = [frontiers, frontiers]
        coordinator = Coordinator(robots, vlm, **parameters)
        coordinator.available_targets = ['A', 'B']
        coordinator.robot_positions = {1: (0, 0), 2: (10, 0)}
        coordinator.run_discussion(['first proposal', 'second proposal'])
        return coordinator

    def run_voting(self, coordinator):
        cot = dict(Meta_Prompt='meta', Frontier_Choice_Prompt='choose', Meta_Choice='A',
                   Choice_Reason='reason', sem_map_frontier=None)
        with patch('src.GDP_agent.random.randint', return_value=2):
            coordinator.run_r2_discussion([cot, cot])

    def test_lambda_alpha_and_robot_specific_positions_reach_bids(self):
        coordinator = self.make_coordinator(decision_lambda=.25, voting_alpha=.7)
        self.run_voting(coordinator)
        self.assertAlmostEqual(coordinator.r3_inputs[1]['A'], .825)
        self.assertAlmostEqual(coordinator.r3_inputs[1]['B'], .175)
        self.assertAlmostEqual(coordinator.r3_inputs[2]['A'], .14)
        self.assertAlmostEqual(coordinator.r3_inputs[2]['B'], .86)
        # lambda controls updated vs original, alpha controls semantics vs distance.
        coordinator.decision_lambda = 1.
        coordinator.voting_alpha = 1.
        self.run_voting(coordinator)
        self.assertEqual(coordinator.r3_inputs[1], {'A': 0., 'B': 1.})
        coordinator.voting_alpha = 0.
        self.run_voting(coordinator)
        self.assertEqual(coordinator.r3_inputs[1], {'A': 1., 'B': 0.})
        self.assertEqual(coordinator.r3_inputs[2], {'A': 0., 'B': 1.})

    def test_settlement_exposes_complete_bundles_payment_and_utility(self):
        coordinator = self.make_coordinator(valuation_scale=.8)
        coordinator.r3_inputs = {1: {'A': .7, 'B': .3}, 2: {'A': .2, 'B': .1}}
        result = coordinator.run_r3_discussion()
        self.assertTrue(result.settled)
        self.assertEqual(coordinator.Final_Bundles, {1: ('A', 'B'), 2: ()})
        self.assertEqual(coordinator.Final_Decision, {1: 'A'})
        self.assertAlmostEqual(result.valuations[1], .8)
        self.assertAlmostEqual(result.payments[1], .24)
        self.assertAlmostEqual(result.net_utilities[1], .56)
        coordinator.valuation_scale = .4
        scaled = coordinator.run_r3_discussion()
        self.assertEqual(scaled.bundles, result.bundles)
        self.assertAlmostEqual(scaled.payments[1], .12)
        self.assertAlmostEqual(scaled.net_utilities[1], .28)

    def test_reserve_is_used_in_actual_and_counterfactual_objective(self):
        coordinator = self.make_coordinator(reserve_price=.5)
        coordinator.r3_inputs = {1: {'A': .7, 'B': .3}, 2: {'A': .2, 'B': .1}}
        result = coordinator.run_r3_discussion()
        self.assertEqual(result.bundles, {1: ('A',), 2: ()})
        self.assertEqual(result.unsold_targets, ('B',))
        self.assertAlmostEqual(result.payments[1], .5)
        self.assertAlmostEqual(result.net_utilities[1], .06)

    def test_new_round_and_history_skip_cannot_reuse_old_settlement(self):
        coordinator = self.make_coordinator()
        self.run_voting(coordinator)
        self.assertIsNotNone(coordinator.run_r3_discussion())
        coordinator.robots[0].navigation_mode = 'history'
        coordinator.run_discussion(['history reason', 'frontier reason'])
        self.assertIsNone(coordinator.auction_result)
        self.assertEqual(coordinator.Final_Bundles, {})
        self.run_voting(coordinator)
        self.assertTrue(coordinator.is_History)
        self.assertEqual(coordinator.r3_inputs, {})
        self.assertIsNone(coordinator.run_r3_discussion())

    def test_invalid_bids_never_produce_a_settled_result(self):
        coordinator = self.make_coordinator()
        coordinator.r3_inputs = {1: {'A': float('nan')}}
        with self.assertRaises(ValueError):
            coordinator.run_r3_discussion()
        self.assertIsNone(coordinator.auction_result)


class VotingTests(unittest.TestCase):
    def test_missing_frontier_does_not_shift_score_indices(self):
        scores = calculate_R3_weighted_scores({1: [.1, .2, .7]}, {1: {'B': 1., 'C': 1.}}, alpha=1.)
        self.assertAlmostEqual(scores[1]['B'], 2 / 9)
        self.assertAlmostEqual(scores[1]['C'], 7 / 9)

    def test_zero_scores_and_zero_distance_have_finite_limits(self):
        scores = calculate_R3_weighted_scores({1: [0., 0.]}, {1: {'A': 0., 'B': 2.}}, alpha=.7)
        self.assertAlmostEqual(scores[1]['A'], .65)
        self.assertAlmostEqual(scores[1]['B'], .35)

    def test_invalid_weights_scores_and_distances_are_rejected(self):
        with self.assertRaises(ValueError):
            optimal_weighted_combination([.5], [.4, .6])
        with self.assertRaises(ValueError):
            optimal_weighted_combination([.5], [.6], weights=(.3, .3))
        for alpha in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                calculate_R3_weighted_scores({1: [1.]}, {1: {'A': 1.}}, alpha)
        for distance in (-1., float('inf')):
            with self.assertRaises(ValueError):
                calculate_R3_weighted_scores({1: [1.]}, {1: {'A': distance}})


if __name__ == '__main__':
    unittest.main()
