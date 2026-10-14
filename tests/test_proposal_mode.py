import sys
import types
import unittest


sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from src.GDP_agent import RobotAgent


class ProposalModeTests(unittest.TestCase):
    def _robot(self):
        robot = RobotAgent(robot_id=1, vlm_models=None)
        robot.target_object = "tv"
        robot.target_point = "A"
        robot.DS = 0.8
        return robot

    def test_explicit_frontier_mode_overrides_low_js(self):
        robot = self._robot()
        robot.JS = 0.1
        robot.navigation_mode = "frontier"
        proposal = robot.propose("reason")
        self.assertEqual(proposal["proposal"]["type"], "frontier")

    def test_explicit_history_mode_overrides_high_js(self):
        robot = self._robot()
        robot.JS = 0.9
        robot.navigation_mode = "history"
        proposal = robot.propose("reason")
        self.assertEqual(proposal["proposal"]["type"], "history")


if __name__ == "__main__":
    unittest.main()
