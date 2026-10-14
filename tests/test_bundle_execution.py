"""Dependency-free tests for committed bundle execution and planner signalling."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from src.bundle_execution import BundleExecution, map_position_xy


class BundleExecutionTests(unittest.TestCase):
    def setUp(self):
        self.execution = BundleExecution()
        self.coordinates = {"A": [10, 20], "B": [30, 40], "C": [0, 0]}

    def test_executes_entire_bundle_in_awarded_order(self):
        self.execution.install({1: ("B", "A"), 2: ("C",)}, self.coordinates, 49)
        self.assertEqual(self.execution.current(1), (30, 40))
        self.assertTrue(self.execution.complete(1, [30, 40]))
        self.assertEqual(self.execution.current(1), (10, 20))
        self.assertTrue(self.execution.complete(1, [10, 20]))
        self.assertIsNone(self.execution.current(1))
        self.assertTrue(self.execution.has_pending())
        self.assertTrue(self.execution.complete(2, (0, 0)))
        self.assertFalse(self.execution.has_pending())
        completed = [event for event in self.execution.drain_events()
                     if event["event"] == "completed"]
        self.assertEqual([event["target"] for event in completed], ["B", "A", "C"])

    def test_relabeling_and_input_mutation_cannot_redirect_queue(self):
        awarded = ["A", "B"]
        self.execution.install({1: awarded}, self.coordinates, 49)
        awarded.reverse()
        self.coordinates["A"][:] = [200, 210]
        self.coordinates["B"] = [400, 410]
        self.assertEqual(self.execution.current(1, 74), (10, 20))
        self.assertTrue(self.execution.complete(1, (10, 20)))
        self.assertEqual(self.execution.current(1, 99), (30, 40))

    def test_stale_completion_signal_does_not_skip_next_target(self):
        self.execution.install({1: ("A", "B")}, self.coordinates, 49)
        self.assertFalse(self.execution.complete(1, (99, 99)))
        self.assertTrue(self.execution.complete(1, (10, 20)))
        self.assertFalse(self.execution.complete(1, (10, 20)))
        self.assertEqual(self.execution.current(1), (30, 40))

    def test_pending_batch_cannot_be_preempted(self):
        self.execution.install({1: ("A", "B")}, self.coordinates, 49)
        with self.assertRaises(RuntimeError):
            self.execution.install({1: ("C",)}, self.coordinates, 74)
        self.assertEqual(self.execution.current(1), (10, 20))
        self.execution.complete(1, (10, 20))
        self.execution.complete(1, (30, 40))
        self.execution.install({1: ("C",)}, self.coordinates, 99)
        self.assertEqual(self.execution.current(1), (0, 0))

    def test_install_validation_is_atomic(self):
        invalid_allocations = [
            ({1: ("A",), 2: ("A",)}, self.coordinates),
            ({1: ("A", "A")}, self.coordinates),
            ({1: ("A",), 2: ("unknown",)}, self.coordinates),
            ({1: ("A",), 2: ("B",)}, {"A": (10, 20), "B": (-1, 30)}),
            ({1: ("A",)}, {"A": (1.5, 30)}),
            ({1: ("A",)}, {"A": (float("nan"), 30)}),
            ({1: ("A",)}, {"A": (True, 30)}),
            ({1: "A"}, self.coordinates),
        ]
        for bundles, coordinates in invalid_allocations:
            with self.subTest(bundles=bundles, coordinates=coordinates):
                with self.assertRaises(ValueError):
                    self.execution.install(bundles, coordinates, 49)
                self.assertFalse(self.execution.has_pending())
                self.assertEqual(self.execution.drain_events(), [])

    def test_empty_bundle_is_waiting_not_a_fallback_goal(self):
        self.execution.install({1: ("A",), 2: ()}, self.coordinates, 49)
        self.assertTrue(self.execution.manages(2))
        self.assertFalse(self.execution.has_robot_pending(2))
        self.assertIsNone(self.execution.current(2))
        self.assertFalse(self.execution.complete(2, (0, 0)))
        self.assertTrue(self.execution.has_pending())

    def test_completed_robot_stays_managed_until_released(self):
        self.execution.install({1: ("A",), 2: ()}, self.coordinates, 49)
        self.execution.complete(1, (10, 20))
        self.assertFalse(self.execution.has_pending())
        self.assertTrue(self.execution.manages(1))
        self.assertTrue(self.execution.manages(2))
        self.execution.drain_events()
        self.assertTrue(self.execution.cancel(1, "history"))
        self.assertFalse(self.execution.manages(1))
        self.assertEqual(self.execution.drain_events()[0]["event"], "released")
        self.execution.clear("batch_finished")
        self.assertFalse(self.execution.manages(2))
        self.assertFalse(self.execution.has_pending())

    def test_recovery_pause_preserves_queue_and_expires_at_boundary(self):
        self.execution.install({1: ("A", "B")}, self.coordinates, 49)
        self.assertTrue(self.execution.pause(1, until_step=74))
        self.assertTrue(self.execution.has_robot_pending(1))
        self.assertIsNone(self.execution.current(1))
        self.assertIsNone(self.execution.current(1, 73))
        self.assertFalse(self.execution.complete(1, (10, 20)))
        self.assertTrue(self.execution.is_paused(1, 73))
        self.assertEqual(self.execution.current(1, 74), (10, 20))
        self.assertFalse(self.execution.is_paused(1, 74))
        self.assertTrue(self.execution.complete(1, (10, 20)))
        self.assertEqual(self.execution.current(1, 75), (30, 40))
        events = self.execution.drain_events()
        self.assertEqual(sum(event["event"] == "resumed" for event in events), 1)

    def test_cancellation_is_distinct_from_completion(self):
        self.execution.install({1: ("A", "B"), 2: ("C",)}, self.coordinates, 49)
        self.execution.drain_events()
        self.execution.complete(1, (10, 20))
        self.execution.pause(1, 74)
        self.assertTrue(self.execution.cancel(1, "history"))
        self.assertFalse(self.execution.manages(1))
        self.assertFalse(self.execution.cancel(1, "history"))
        self.assertIsNone(self.execution.current(1, 74))
        self.execution.clear("object_goal")
        self.assertFalse(self.execution.has_pending())
        self.assertFalse(self.execution.manages(2))
        events = self.execution.drain_events()
        completed = [event for event in events if event["event"] == "completed"]
        cancelled = [event for event in events if event["event"] == "cancelled"]
        self.assertEqual([event["target"] for event in completed], ["A"])
        self.assertEqual([event["targets"] for event in cancelled], [["B"], ["C"]])
        self.assertEqual([event["reason"] for event in cancelled], ["history", "object_goal"])
        self.assertEqual(self.execution.drain_events(), [])


class MapPositionTests(unittest.TestCase):
    def test_returns_column_row_using_correct_window_offsets(self):
        pose = (3.0, 2.0, 90.0, 10, 90, 20, 100)
        self.assertEqual(map_position_xy(pose, 5, (80, 80)), (40, 30))

    def test_clips_boundaries_and_keeps_zero_cells(self):
        self.assertEqual(map_position_xy((-1, -1, 0, 0, 80, 0, 100), 5, (80, 100)), (0, 0))
        self.assertEqual(map_position_xy((50, 50, 0, 0, 80, 0, 100), 5, (80, 100)), (99, 79))

    def test_invalid_resolution_and_shape_are_rejected(self):
        pose = (3.0, 2.0, 90.0, 10, 90, 20, 100)
        for resolution in (0, -5, float("nan"), float("inf"), True):
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                map_position_xy(pose, resolution, (80, 80))
        with self.assertRaises(ValueError):
            map_position_xy(pose, 5, (0, 80))


class PlannerSignalTests(unittest.TestCase):
    """Exercise the signal hooks without loading Habitat, Torch, or model files."""

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "agents" / "vlm_agents.py"
        tree = ast.parse(path.read_text())
        cls.methods = {
            node.name: node
            for class_node in tree.body if isinstance(class_node, ast.ClassDef)
            for node in class_node.body if isinstance(node, ast.FunctionDef)
        }

    def test_act_resets_signal_before_processing_the_current_goal(self):
        first_statements = self.methods["act"].body[:2]
        robot = SimpleNamespace(navigation_goal_reached=True, navigation_goal_xy=(0, 0))
        scope = {"self": robot, "goal_points": [10, 20]}
        module = ast.Module(body=first_statements, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "<act signal reset>", "exec"), scope)
        self.assertFalse(robot.navigation_goal_reached)
        self.assertEqual(robot.navigation_goal_xy, (10, 20))
        scope["goal_points"][:] = [30, 40]
        self.assertEqual(robot.navigation_goal_xy, (10, 20))

    def test_fmm_stop_marks_frontier_arrival_but_not_object_arrival(self):
        assignments = [
            node for node in ast.walk(self.methods["_plan"])
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) and target.attr == "navigation_goal_reached"
                for target in node.targets
            )
        ]
        signal_assignments = [node for node in assignments if isinstance(node.value, ast.Call)]
        self.assertEqual(len(signal_assignments), 1)
        module = ast.Module(body=signal_assignments, type_ignores=[])
        hook = compile(ast.fix_missing_locations(module), "<FMM completion signal>", "exec")
        for stop, found_goal, expected in ((False, 0, False), (True, 0, True),
                                           (False, 1, False), (True, 1, False)):
            with self.subTest(stop=stop, found_goal=found_goal):
                robot = SimpleNamespace(navigation_goal_reached=False)
                exec(hook, {"self": robot, "stop": stop, "planner_inputs": {"found_goal": found_goal}})
                self.assertEqual(robot.navigation_goal_reached, expected)

    def test_waiting_scans_in_place_without_overriding_a_detected_object(self):
        waiting_branches = [
            node for node in self.methods["_plan"].body
            if isinstance(node, ast.If) and any(
                isinstance(child, ast.Constant) and child.value == "wait_for_allocation"
                for child in ast.walk(node.test)
            )
        ]
        self.assertEqual(len(waiting_branches), 1)
        function = ast.parse("def waiting_action(self, planner_inputs):\n    return None\n").body[0]
        function.body.insert(0, waiting_branches[0])
        scope = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     "<waiting action>", "exec"), scope)
        for waiting, found_goal, expected in ((True, 0, 3), (True, 1, None),
                                              (False, 0, None), (False, 1, None)):
            with self.subTest(waiting=waiting, found_goal=found_goal):
                robot = SimpleNamespace(navigation_goal_reached=False)
                action = scope["waiting_action"](robot, {
                    "wait_for_allocation": waiting, "found_goal": found_goal,
                })
                self.assertEqual(action, expected)
                self.assertFalse(robot.navigation_goal_reached)

        fmm_calls = [node for node in ast.walk(self.methods["_plan"])
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "_get_stg"]
        self.assertEqual(len(fmm_calls), 1)
        self.assertLess(waiting_branches[0].lineno, fmm_calls[0].lineno)

    def test_act_forwards_waiting_flag_to_the_local_planner(self):
        assignments = [
            node for node in ast.walk(self.methods["act"])
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Subscript)
                and any(isinstance(child, ast.Constant) and child.value == "wait_for_allocation"
                        for child in ast.walk(target))
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        module = ast.Module(body=assignments, type_ignores=[])
        forward = compile(ast.fix_missing_locations(module), "<forward wait flag>", "exec")
        for waiting in (True, False):
            planner_inputs = {}
            exec(forward, {"planner_inputs": planner_inputs, "wait_for_allocation": waiting})
            self.assertIs(planner_inputs["wait_for_allocation"], waiting)


if __name__ == "__main__":
    unittest.main()
