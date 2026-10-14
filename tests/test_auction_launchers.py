"""Check experiment labels against the parameters actually passed to Python."""

from pathlib import Path
import re
import shlex
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AuctionLauncherTests(unittest.TestCase):
    def test_lambda_alpha_launchers_pass_the_labelled_parameter(self):
        launchers = sorted(ROOT.glob("tpami_mrdn_lambda*.sh"))
        launchers += sorted(ROOT.glob("tpami_mrdn_alpha*.sh"))
        self.assertEqual(len(launchers), 6)
        for path in launchers:
            with self.subTest(launcher=path.name):
                command = shlex.split(path.read_text().replace("\\\n", ""), comments=True)
                self.assertEqual(command[:2], ["python", "main_tpami_250903.py"])
                flags = command[2:]
                self.assertEqual(len(flags) % 2, 0)
                arguments = dict(zip(flags[::2], flags[1::2]))
                self.assertEqual(len(arguments) * 2, len(flags), "duplicate CLI parameter")
                variant = re.fullmatch(r"tpami_mrdn_(lambda|alpha)([0-9.]+)_([0-9]+)\.sh", path.name)
                self.assertIsNotNone(variant)
                parameter, value, date = variant.groups()
                expected_lambda = value if parameter == "lambda" else "0.3"
                expected_alpha = value if parameter == "alpha" else "0.7"
                self.assertEqual(arguments["--consensus_lambda"], expected_lambda)
                self.assertEqual(arguments["--voting_alpha"], expected_alpha)
                self.assertEqual(arguments["--auction_beta"], "0.8")
                self.assertNotIn("--alpha", arguments, "--alpha belongs to the RL optimizer")
                self.assertEqual(arguments["-d"],
                                 "./VLM_EXP/multi_hm3d_2-robot_tpami_{}{}_{}/".format(parameter, value, date))
                self.assertEqual(arguments["--num_agents"], "2")
                self.assertEqual(arguments["--visualize"], "2")
                self.assertEqual(arguments["--num_sem_categories"], "16")
                self.assertEqual(arguments["--task_config"], "tasks/multi_objectnav_hm3d.yaml")
                self.assertEqual(arguments["--base_url"], "http://127.0.0.1:31516")


if __name__ == "__main__":
    unittest.main()
