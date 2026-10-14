"""Check the heavyweight entrypoint integration without importing Habitat."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AuctionEntrypointTests(unittest.TestCase):
    def entrypoints(self):
        for name in ('main_tpami_250903.py', 'review_main_final_260217.py'):
            yield name, ast.parse((ROOT / name).read_text())

    def test_decision_prompt_uses_full_label_space(self):
        # The formatter enumerates its dictionary as A--D. Passing a filtered
        # dictionary would renumber labels and break the full-map score indices.
        for name, tree in self.entrypoints():
            with self.subTest(entrypoint=name):
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name)
                         and node.func.id == 'form_prompt_for_DecisionVLM_Frontier']
                self.assertEqual(len(calls), 2)
                for call in calls:
                    self.assertIsInstance(call.args[5], ast.Name)
                    self.assertEqual(call.args[5].id, 'full_Frontiers_dict')

    def test_active_and_gt_planner_interfaces_are_not_confused(self):
        for name, tree in self.entrypoints():
            with self.subTest(entrypoint=name):
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr == 'act'
                         and isinstance(node.func.value, ast.Subscript)]
                actual = [node for node in calls if node.func.value.value.id == 'agent']
                gt = [node for node in calls if node.func.value.value.id == 'agent_GT']
                self.assertEqual(len(actual), 1)
                self.assertEqual(len(gt), 1)
                self.assertIn('wait_for_allocation', [k.arg for k in actual[0].keywords])
                self.assertNotIn('wait_for_allocation', [k.arg for k in gt[0].keywords])

    def test_coordinator_cli_and_complete_bundle_install_are_wired(self):
        for name, tree in self.entrypoints():
            with self.subTest(entrypoint=name):
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name)]
                constructor = next(node for node in calls if node.func.id == 'Coordinator')
                values = {k.arg: ast.unparse(k.value) for k in constructor.keywords}
                self.assertEqual(values['decision_lambda'], 'args.consensus_lambda')
                self.assertEqual(values['voting_alpha'], 'args.voting_alpha')
                self.assertEqual(values['valuation_scale'], 'args.auction_beta')
                self.assertEqual(values['reserve_price'], 'args.auction_reserve_price')
                installs = [node for node in calls if node.func.id == 'install_settled_bundles']
                self.assertEqual(len(installs), 1)
                self.assertEqual(ast.unparse(installs[0].args[1]), 'coordinator.auction_result')
                self.assertFalse(any(node.func.id in ('exit', 'quit') for node in calls))


if __name__ == '__main__':
    unittest.main()
