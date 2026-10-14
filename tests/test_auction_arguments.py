"""Test the real CLI parser without importing the GPU dependency."""

import argparse
import ast
from contextlib import redirect_stderr
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class AuctionArgumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'arguments.py'
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'get_args')
        scope = {'argparse': argparse, 'math': math,
                 'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
        cls.parse_args = staticmethod(scope['get_args'])

    def parse(self, *args):
        with patch.object(sys, 'argv', ['navigation', '--no_cuda', *args]):
            return self.parse_args()

    def test_defaults_and_optimizer_alpha_are_separate(self):
        args = self.parse()
        self.assertEqual((args.consensus_lambda, args.voting_alpha, args.auction_beta,
                          args.auction_reserve_price), (.3, .7, .8, 0.))
        self.assertEqual(args.alpha, .99)

    def test_explicit_parameters_are_used(self):
        args = self.parse('--consensus_lambda', '.5', '--voting_alpha', '.9',
                          '--auction_beta', '.6', '--auction_reserve_price', '.1')
        self.assertEqual((args.consensus_lambda, args.voting_alpha, args.auction_beta,
                          args.auction_reserve_price), (.5, .9, .6, .1))

    def test_invalid_auction_parameters_fail_at_parse_time(self):
        invalid = [('--consensus_lambda', '1.1'), ('--voting_alpha', 'nan'),
                   ('--auction_beta', '0'), ('--auction_beta', 'inf'),
                   ('--auction_reserve_price', '-1'), ('--num_local_steps', '0')]
        for flag, value in invalid:
            with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse(flag, value)


if __name__ == '__main__':
    unittest.main()
