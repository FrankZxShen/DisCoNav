import itertools
import random
import unittest
from dataclasses import replace

from src.target_auction import PriorityAuctionSystem


def brute_force_welfare(robots, targets, bids, scale, capacity, reserve, excluded=()):
    """Independent oracle: enumerate every target's owner or seller retention."""
    robots = tuple(robot for robot in robots if robot not in excluded)
    best = 0.0
    for owners in itertools.product((None,) + robots, repeat=len(targets)):
        counts = {robot: 0 for robot in robots}
        score = 0.0
        valid = True
        for target, owner in zip(targets, owners):
            if owner is None:
                score += reserve
            elif target not in bids.get(owner, {}):
                valid = False
                break
            else:
                counts[owner] += 1
                score += scale * bids[owner][target]
        if not valid or (capacity is not None and any(count > capacity for count in counts.values())):
            continue
        best = max(best, score)
    return best


class TargetAuctionTests(unittest.TestCase):
    def test_normalized_votes_produce_multi_target_bundle_and_pivot_payments(self):
        auction = PriorityAuctionSystem([1, 2], ['A', 'B', 'C'], {
            1: {'A': .6, 'B': .3, 'C': .1}, 2: {'A': .2, 'B': .1, 'C': .7}})
        auction.auction_process()
        result = auction.get_settled_result()
        self.assertEqual(result.bundles, {1: ('A', 'B'), 2: ('C',)})
        self.assertAlmostEqual(result.welfare, 1.28)
        self.assertAlmostEqual(result.payments[1], .24)
        self.assertAlmostEqual(result.payments[2], .08)
        self.assertAlmostEqual(result.net_utilities[1], .48)
        self.assertAlmostEqual(result.net_utilities[2], .48)

    def test_exact_allocation_beats_old_bundle_greedy(self):
        auction = PriorityAuctionSystem([1, 2], ["A", "B"], {
            1: {"A": 0.7, "B": 0.3}, 2: {"A": 0.4, "B": 0.6}})
        self.assertEqual(auction.auction_process(), {1: "A", 2: "B"})
        result = auction.get_settled_result()
        self.assertEqual(result.bundles, {1: ("A",), 2: ("B",)})
        self.assertAlmostEqual(result.welfare, 1.04)
        self.assertAlmostEqual(result.payments[1], 0.32)
        self.assertAlmostEqual(result.payments[2], 0.24)
        self.assertAlmostEqual(result.net_utilities[1], 0.24)
        self.assertAlmostEqual(result.net_utilities[2], 0.24)
        self.assertTrue(result.settled)

    def test_full_multi_target_bundle_is_retained(self):
        auction = PriorityAuctionSystem([1, 2], ["A", "B"], {
            1: {"A": 0.9, "B": 0.8}, 2: {"A": 0.4, "B": 0.3}})
        self.assertEqual(auction.auction_process(), {1: "A"})
        result = auction.get_settled_result()
        self.assertEqual(result.bundles[1], ("A", "B"))
        self.assertEqual(result.bundles[2], ())
        self.assertEqual(auction.assignments, {1: ("A", "B")})
        self.assertAlmostEqual(result.payments[1], 0.56)
        self.assertAlmostEqual(result.net_utilities[1], 0.8)
        self.assertEqual(result.payments[2], 0.0)
        self.assertEqual(result.unsold_targets, ())

    def test_capacity_restricts_bundle_inside_solver(self):
        auction = PriorityAuctionSystem([1, 2], ["A", "B"], {
            1: {"A": 0.9, "B": 0.8}, 2: {"A": 0.7, "B": 0.1}},
            valuation_scale=1.0, max_targets_per_robot=1)
        self.assertEqual(auction.auction_process(), {1: "B", 2: "A"})
        self.assertAlmostEqual(auction.result.welfare, 1.5)
        self.assertTrue(all(len(bundle) <= 1 for bundle in auction.result.bundles.values()))

    def test_single_robot_counterfactual_has_zero_robot_welfare(self):
        auction = PriorityAuctionSystem([1], ["A", "B"], {1: {"A": 0.7, "B": 0.3}})
        auction.auction_process()
        self.assertEqual(auction.result.bundles[1], ("A", "B"))
        self.assertEqual(auction.solve_winner_determination([1]), 0.0)
        self.assertEqual(auction.result.payments[1], 0.0)
        self.assertAlmostEqual(auction.result.net_utilities[1], 0.8)

    def test_reserve_is_seller_value_in_both_actual_and_counterfactual(self):
        auction = PriorityAuctionSystem([1], ["A", "B", "C"], {
            1: {"A": 0.7, "B": 0.3, "C": 0.05}}, reserve_price=0.1)
        auction.auction_process()
        result = auction.get_settled_result()
        self.assertEqual(result.bundles[1], ("A", "B"))
        self.assertEqual(result.unsold_targets, ("C",))
        self.assertAlmostEqual(result.welfare, 0.9)
        self.assertAlmostEqual(result.seller_value, 0.1)
        self.assertAlmostEqual(result.counterfactual_welfare[1], 0.3)
        self.assertAlmostEqual(result.payments[1], 0.2)
        self.assertAlmostEqual(result.net_utilities[1], 0.6)

    def test_uniform_scale_applies_to_singletons_bundles_and_payments(self):
        bids = {1: {"A": 0.7, "B": 0.3}, 2: {"A": 0.4, "B": 0.6}}
        full = PriorityAuctionSystem([1, 2], ["A", "B"], bids, valuation_scale=1.0)
        scaled = PriorityAuctionSystem([1, 2], ["A", "B"], bids, valuation_scale=0.8)
        self.assertEqual(full.auction_process(), scaled.auction_process())
        self.assertAlmostEqual(scaled.bundle_valuations[(1, ("A",))], 0.56)
        self.assertAlmostEqual(scaled.bundle_valuations[(1, ("A", "B"))], 0.8)
        for robot in (1, 2):
            self.assertAlmostEqual(scaled.payments[robot], 0.8 * full.payments[robot])
            self.assertAlmostEqual(scaled.net_utilities[robot], 0.8 * full.net_utilities[robot])

    def test_payments_do_not_rerank_the_allocated_targets(self):
        auction = PriorityAuctionSystem([1, 2], ["A", "B"], {
            1: {"A": 1.0, "B": 0.1}, 2: {"A": 0.99, "B": 0.8}},
            valuation_scale=1.0)
        self.assertEqual(auction.auction_process(), {1: "A", 2: "B"})
        self.assertLess(auction.net_utilities[1], auction.net_utilities[2])
        self.assertEqual(auction.result.bundles[1], ("A",))

    def test_missing_edges_never_create_bids(self):
        auction = PriorityAuctionSystem([1, 2], ["A", "B", "C"], {1: {"A": 0.6}})
        self.assertEqual(auction.auction_process(), {1: "A"})
        self.assertEqual(auction.result.bundles[2], ())
        self.assertEqual(auction.result.unsold_targets, ("B", "C"))

    def test_empty_robots_targets_and_zero_surplus(self):
        for robots, targets, bids in (([], ["A"], {}), ([1], [], {1: {}}),
                                      ([1], ["A"], {1: {"A": 0.0}}), ([], [], {})):
            with self.subTest(robots=robots, targets=targets):
                auction = PriorityAuctionSystem(robots, targets, bids)
                self.assertEqual(auction.auction_process(), {})
                self.assertEqual(auction.result.unsold_targets, tuple(targets))
                self.assertTrue(auction.result.settled)
        zero_capacity = PriorityAuctionSystem([1], ["A"], {1: {"A": 1}}, max_targets_per_robot=0)
        self.assertEqual(zero_capacity.auction_process(), {})

    def test_ties_do_not_depend_on_input_order(self):
        forward = PriorityAuctionSystem([1, 2], ["A", "B"], {
            1: {"A": 0.5, "B": 0.5}, 2: {"A": 0.5, "B": 0.5}}, max_targets_per_robot=1)
        reverse = PriorityAuctionSystem([2, 1], ["B", "A"], {
            2: {"B": 0.5, "A": 0.5}, 1: {"B": 0.5, "A": 0.5}}, max_targets_per_robot=1)
        self.assertEqual(forward.auction_process(), {1: "A", 2: "B"})
        self.assertEqual(reverse.auction_process(), forward.auction_process())

    def test_repeated_auctions_reset_settlement(self):
        auction = PriorityAuctionSystem([1], ["A"], {1: {"A": 0.9}})
        first = auction.auction_process()
        auction.payments[1] = 999
        self.assertEqual(auction.auction_process(), first)
        self.assertEqual(auction.payments[1], 0.0)
        self.assertTrue(auction.get_settled_result().settled)
        self.assertEqual(auction.calculate_vcg_payment(1, ("A",)), 0.0)
        with self.assertRaises(ValueError):
            auction.calculate_vcg_payment(1, ())

    def test_execution_requires_valid_settlement(self):
        auction = PriorityAuctionSystem([1], ["A"], {1: {"A": 1.0}})
        with self.assertRaises(ValueError):
            auction.get_settled_result()
        auction.auction_process()
        corrupted = replace(auction.result, payments={1: 10.0})
        with self.assertRaises(ValueError):
            auction.validate_settlement(corrupted)
        with self.assertRaises(TypeError):
            auction.result.payments[1] = 10.0
        self.assertTrue(auction.result.as_dict()["settled"])

    def test_invalid_inputs(self):
        cases = [
            (([1, 1], ["A"], {}), {}),
            (([1], ["A", "A"], {}), {}),
            (([1], ["A"], {2: {"A": 1}}), {}),
            (([1], ["A"], {1: {"B": 1}}), {}),
            (([1], ["A"], {1: {"A": -1}}), {}),
            (([1], ["A"], {1: {"A": float("nan")}}), {}),
            (([1], ["A"], {1: {"A": float("inf")}}), {}),
            (([1], ["A"], {1: {"A": True}}), {}),
            (([1], ["A"], {}), {"valuation_scale": -0.1}),
            (([1], ["A"], {}), {"reserve_price": float("inf")}),
            (([1], ["A"], {}), {"max_targets_per_robot": 1.5}),
            (([1], ["A"], {}), {"max_targets_per_robot": -1}),
            (([1], ["A"], {}), {"max_targets_per_robot": True}),
        ]
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                PriorityAuctionSystem(*args, **kwargs)

    def test_random_small_instances_against_independent_oracle(self):
        generator = random.Random(42)
        for case in range(100):
            robots = tuple(range(1, generator.randrange(1, 4) + 1))
            targets = tuple("ABCD"[:generator.randrange(0, 5)])
            bids = {robot: {target: generator.randrange(0, 11) / 10
                            for target in targets if generator.random() > 0.15}
                    for robot in robots}
            scale = generator.choice((0.0, 0.5, 0.8, 1.0))
            capacity = generator.choice((None, 0, 1, 2))
            reserve = generator.choice((0.0, 0.1, 0.3))
            with self.subTest(case=case):
                auction = PriorityAuctionSystem(robots, targets, bids, scale, capacity, reserve)
                auction.auction_process()
                result = auction.get_settled_result()
                expected = brute_force_welfare(robots, targets, bids, scale, capacity, reserve)
                self.assertAlmostEqual(result.welfare, expected)
                allocated = [target for bundle in result.bundles.values() for target in bundle]
                self.assertEqual(len(allocated), len(set(allocated)))
                for robot in robots:
                    excluded = brute_force_welfare(robots, targets, bids, scale, capacity, reserve, (robot,))
                    self.assertAlmostEqual(result.counterfactual_welfare[robot], excluded)
                    self.assertAlmostEqual(result.payments[robot],
                                           excluded - (expected - result.valuations[robot]))
                    self.assertGreaterEqual(result.payments[robot], -1e-10)
                    self.assertGreaterEqual(result.net_utilities[robot], -1e-10)


if __name__ == "__main__":
    unittest.main()
