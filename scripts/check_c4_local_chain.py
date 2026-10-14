"""A deterministic C4 interface smoke check, NOT a Habitat navigation run.

Run from the repository root: python scripts/check_c4_local_chain.py
VLM responses, robot poses and short-term-goal/arrival decisions are fixtures.
Voting, exact allocation, payment, queue management, act and _plan are real
repository implementations. act/_plan are loaded unchanged through AST to
avoid importing the unavailable GPU/Habitat stack. No production file is edited.
"""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Dict
from unittest.mock import patch

if not __debug__:
    raise RuntimeError('Run without -O/PYTHONOPTIMIZE: smoke-check assertions must be enabled.')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEPENDENCIES = {name: importlib.util.find_spec(name) is not None
                for name in ('numpy', 'torch', 'habitat', 'cv2', 'skfmm', 'skimage')}

import numpy as np

# GDP imports cv2, but this smoke check never calls a cv2 operation.
if not DEPENDENCIES['cv2']:
    sys.modules['cv2'] = ModuleType('cv2')
from src.GDP_agent import Coordinator, RobotAgent
from src.auction_navigation import (
    consume_completed_goals, execution_goals, frontier_coordinates,
    install_settled_bundles, select_frontier_proposal,
)
from src.bundle_execution import BundleExecution
from src.target_auction import PriorityAuctionSystem
from utils import pose as pu


class FixtureVLM:
    """Known outputs, without a model server or semantic-inference claim."""
    def COT3(self, **kwargs):
        if kwargs['return_string_probabilities'] is None:
            return None, json.dumps({
                'rebuttal': {'rebuttal_reason': 'fixture critique'},
                'required_adjustment': {'new_target': 'A',
                                        'recommendation': 'fixture recommendation'},
            })
        return [.3, .1, .1, .5], 'D'


def load_functions(path, names, scope, class_name=None):
    """Execute the exact saved function bodies, with explicit test dependencies."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    body = tree.body
    if class_name:
        body = next(node for node in body if isinstance(node, ast.ClassDef)
                    and node.name == class_name).body
    nodes = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return {name: scope[name] for name in names}


class ArrayTensor(np.ndarray):
    """Only the CPU array interface used by act; no Torch computation is claimed."""
    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def local_robot_type():
    drawing = load_functions(ROOT / 'utils/visualization.py', ['draw_line'], {'np': np})
    scope = {'np': np, 'math': math, 'pu': pu, 'vu': SimpleNamespace(**drawing), 'Dict': Dict}
    methods = load_functions(ROOT / 'agents/vlm_agents.py', ['act', '_plan'],
                             scope, class_name='LLM_Agent')

    def fixture_stg(self, grid, start, goal, planning_window):
        # This replaces _get_stg/FMM. Stop derives from the prescribed current
        # position; the smoke check never sets navigation_goal_reached directly.
        row, column = np.argwhere(goal == 1)[0]
        return (int(row), int(column)), tuple(start) == (int(row), int(column))

    return type('LocalRobotFixture', (), {**methods, '_get_stg': fixture_stg})


def make_local_robot(robot_type, xy):
    robot = robot_type()
    robot.args = SimpleNamespace(task_config='objectnav_mp3d', num_local_steps=25,
                                 map_resolution=5, turn_angle=30,
                                 collision_threshold=.2, visualize=0, print_images=0)
    robot.goal_id = 0
    robot.local_w = robot.local_h = 96
    robot.local_map = np.zeros((20, 96, 96)).view(ArrayTensor)
    robot.local_map[1] = 1  # Fully explored, obstacle-free synthetic map.
    robot.target_point_map = np.zeros((96, 96))
    robot.visited = np.zeros((96, 96))
    robot.visited_vis = np.zeros((96, 96))
    robot.collision_map = np.zeros((96, 96))
    robot.col_width = 1
    robot.collision_n = 0
    robot.eve_angle = 0
    robot.Path_Length = 0.
    robot.l_step = 49
    robot.last_action = None
    robot.curr_loc = [xy[0] * .05, xy[1] * .05, 0.]
    robot.planner_pose_inputs = [*robot.curr_loc, 0, 96, 0, 96]
    robot.navigation_goal_reached = False
    robot.navigation_goal_xy = None
    return robot


def main():
    frontiers = {
        'frontier_0': '<centroid: (20, 20), number: 8>',
        'frontier_1': '<centroid: (20, 60), number: 8>',
        'frontier_2': '<centroid: (60, 60), number: 8>',
        'frontier_3': '<centroid: (60, 20), number: 8>',
    }
    unchanged_frontiers = deepcopy(frontiers)
    remaining = dict(frontiers)
    origins = {1: [.1, .1, .1, .7], 2: [.1, .6, .2, .1]}
    positions = {1: (60, 10), 2: (20, 70)}
    proposals, proposal_goals = {}, []
    vlm = FixtureVLM()
    robots = []
    for robot_id in (1, 2):
        label, xy = select_frontier_proposal(frontiers, remaining, origins[robot_id])
        proposals[robot_id] = label
        proposal_goals.append(list(xy))
        robot = RobotAgent(robot_id, vlm)
        robot.navigation_mode = 'frontier'
        robot.target_object = 'chair'
        robot.target_point = label
        robot.JS = .8
        robot.DS = origins[robot_id][ord(label) - ord('A')]
        robot.origin_Decision_PR = origins[robot_id]
        robot.frontier_coords = [frontiers, frontiers]
        robots.append(robot)
    assert proposals == {1: 'D', 2: 'B'}
    assert set(remaining) == {'frontier_0', 'frontier_2'}
    assert frontiers == unchanged_frontiers

    coordinator = Coordinator(robots, vlm, decision_lambda=.3, voting_alpha=.7,
                              valuation_scale=.8, reserve_price=0.)
    coordinator.available_targets = ['A', 'B', 'C', 'D']
    coordinator.robot_positions = positions
    coordinator.run_discussion(['fixture reason for D', 'fixture reason for B'])
    cot = dict(Meta_Prompt='fixture meta', Frontier_Choice_Prompt='fixture choice',
               Meta_Choice='D', Choice_Reason='fixture reason', sem_map_frontier=None)
    with patch('src.GDP_agent.random.randint', return_value=2):
        coordinator.run_r2_discussion([cot, dict(cot, Meta_Choice='B')])
    result = coordinator.run_r3_discussion()
    assert result.settled and result.unsold_targets == ()
    assert result.bundles == {1: ('A', 'D'), 2: ('B', 'C')}
    assert all(result.payments[i] > 0 and result.net_utilities[i] >= 0 for i in (1, 2))
    assert frontiers == unchanged_frontiers

    execution = BundleExecution()
    install_settled_bundles(execution, result, frontiers, coordinator.round)
    local_type = local_robot_type()
    local_robots = [make_local_robot(local_type, positions[i]) for i in (1, 2)]
    trace = []
    prescribed_steps = [
        ('before_first_arrival', {1: (60, 10), 2: (20, 70)}),
        ('at_first_waypoints', {1: (20, 20), 2: (20, 60)}),
        ('second_waypoints_submitted', {1: (20, 20), 2: (20, 60)}),
        ('at_second_waypoints', {1: (60, 20), 2: (60, 60)}),
        ('completed_robots_wait', {1: (60, 20), 2: (60, 60)}),
    ]
    for offset, (stage, stage_positions) in enumerate(prescribed_steps):
        consume_completed_goals(execution, local_robots)
        goals = execution_goals(execution, proposal_goals, stage_positions, 49 + offset)
        row = {'stage': stage, 'pose_source': 'prescribed_fixture',
               'robot_positions_xy': stage_positions, 'submitted_goals_xy': goals,
               'robots': []}
        for i, local in enumerate(local_robots, 1):
            x, y = stage_positions[i]
            local.planner_pose_inputs[:3] = [x * .05, y * .05, 0.]
            waiting = execution.manages(i) and not execution.has_robot_pending(i)
            action = local.act(goals[i - 1], wait_for_allocation=waiting)
            row['robots'].append({'robot_id': i, 'action': int(action), 'waiting': waiting,
                                  'goal_received_by_act': local.navigation_goal_xy,
                                  'navigation_goal_reached': bool(local.navigation_goal_reached)})
        trace.append(row)
    events = execution.drain_events()
    completed = [(e['robot_id'], e['target']) for e in events if e['event'] == 'completed']
    assert completed == [(1, 'A'), (2, 'B'), (1, 'D'), (2, 'C')]
    assert trace[2]['submitted_goals_xy'] == [[60, 20], [60, 60]]
    assert not execution.has_pending()
    assert all(r['action'] == 3 and r['waiting'] for r in trace[-1]['robots'])
    assert all(r['action'] != 0 for row in trace for r in row['robots'])

    # Deliberate boundary input: only B and D have any bids. This is NOT the
    # preceding real voting output and cannot explain the original Fig. 6.
    sparse_bids = {i: {label: value for label, value in bids.items() if label in ('B', 'D')}
                   for i, bids in coordinator.r3_inputs.items()}
    sparse = PriorityAuctionSystem([1, 2], ['A', 'B', 'C', 'D'], sparse_bids)
    sparse.auction_process()
    sparse_result = sparse.get_settled_result()
    assert sparse_result.unsold_targets == ('A', 'C')
    assert frontiers == unchanged_frontiers

    source_paths = ['src/GDP_agent.py', 'src/target_auction.py', 'src/bundle_execution.py',
                    'src/auction_navigation.py', 'agents/vlm_agents.py',
                    'main_tpami_250903.py', 'review_main_final_260217.py']
    return {
        'status': 'PASS', 'kind': 'local_deterministic_interface_smoke',
        'dependencies_available': DEPENDENCIES,
        'fixtures': ['VLM output', 'semantic map', 'robot poses', '_get_stg/FMM result'],
        'real_execution': ['R1/R2/R3 Coordinator', 'voting', 'exact WDP', 'VCG settlement',
                           'BundleExecution', 'LLM_Agent.act', 'LLM_Agent._plan'],
        'not_executed': ['Habitat env.step', 'model inference', 'FMM distance solver',
                         'Frontiers map extraction', 'full main loop', 'performance evaluation'],
        'parameters': {'lambda': .3, 'alpha': .7, 'beta': .8, 'reserve': 0.},
        'inputs': {'frontiers': frontiers, 'positions_xy': positions, 'initial_scores': origins,
                   'robot_1_updated_scores': [.3, .1, .1, .5], 'refuter_robot': 2},
        'initial_proposals': proposals,
        'initial_copy_remaining_labels': ['A', 'C'],
        'full_candidate_pool_unchanged': frontiers == unchanged_frontiers,
        'voting_scores': coordinator.r3_inputs, 'settlement': result.as_dict(),
        'navigation_interface_trace': trace, 'queue_events': events,
        'boundary_case_missing_bids': {
            'synthetic_input_change': 'remove A/C bids from both robots',
            'bids': sparse_bids, 'settlement': sparse_result.as_dict(),
            'original_figure_explanation_verified': False,
        },
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in source_paths},
    }


if __name__ == '__main__':
    captured = io.StringIO()
    with redirect_stdout(captured):
        report = main()
    report['captured_source_stdout'] = captured.getvalue()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
