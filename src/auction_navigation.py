"""Map a settled auction to frozen-coordinate navigation queues."""

import ast
import math


def frontier_coordinates(frontiers):
    """Use the same numeric-key order as the map's A--D frontier labels."""
    keys = sorted(frontiers, key=lambda key: int(key.split('_')[1]))
    coordinates = {}
    seen = set()
    for index, key in enumerate(keys):
        value = frontiers[key]
        if isinstance(value, str):
            value = ast.literal_eval(value.split('centroid: ', 1)[1].split(', number:', 1)[0])
        xy = tuple(value)
        if len(xy) != 2 or any(isinstance(v, bool) or int(v) != v or v < 0 for v in xy):
            raise ValueError('frontier coordinates must be nonnegative integer (x, y) pairs')
        xy = tuple(int(v) for v in xy)
        if xy in seen:
            raise ValueError('distinct frontier labels cannot reserve the same coordinate')
        seen.add(xy)
        coordinates[chr(65 + index)] = xy
    return coordinates


def install_settled_bundles(execution, result, frontiers, round_id):
    """Install all bundles together, after the Coordinator validates settlement."""
    if result is None or not result.settled:
        raise ValueError('only a settled VCG allocation can enter navigation')
    execution.install(result.bundles, frontier_coordinates(frontiers), round_id)


def select_frontier_proposal(full_frontiers, remaining_pool, decision_scores):
    """Choose a preliminary frontier without shifting its original label.

    Scores index the full, numeric-key-ordered A--D map labels even after other
    robots remove candidates.  An exhausted preliminary pool permits a shared
    proposal from the full pool; the subsequent auction resolves conflicts.
    """
    keys = sorted(full_frontiers, key=lambda key: int(key.split('_')[1]))
    if not keys:
        raise ValueError('frontier proposals require at least one candidate')
    if any(key not in full_frontiers for key in remaining_pool):
        raise ValueError('remaining candidates must belong to the full frontier pool')
    if isinstance(decision_scores, (str, bytes)) or len(decision_scores) < len(keys):
        raise ValueError('decision scores must cover every original frontier label')
    scores = [float(value) for value in decision_scores[:len(keys)]]
    if any(not math.isfinite(value) for value in scores):
        raise ValueError('frontier decision scores must be finite')
    coordinates = frontier_coordinates(full_frontiers)
    candidate_indices = [index for index, key in enumerate(keys) if key in remaining_pool]
    if not candidate_indices:
        candidate_indices = list(range(len(keys)))
    choice = max(candidate_indices, key=lambda index: scores[index])
    label = chr(65 + choice)
    remaining_pool.pop(keys[choice], None)
    return label, coordinates[label]


def consume_completed_goals(execution, navigation_robots):
    """Consume a local-planner completion once, matched to its actual waypoint."""
    for robot_id, robot in enumerate(navigation_robots, start=1):
        if getattr(robot, 'navigation_goal_reached', False):
            execution.complete(robot_id, robot.navigation_goal_xy)
            robot.navigation_goal_reached = False


def execution_goals(execution, proposed_goals, positions, step):
    """Keep reservations, let paused robots recover, and hold finished robots."""
    goals = list(proposed_goals)
    for robot_id in positions:
        if not execution.manages(robot_id):
            continue
        waypoint = execution.current(robot_id, step=step)
        if waypoint is not None:
            goals[robot_id - 1] = list(waypoint)
        elif not execution.has_robot_pending(robot_id):
            # No assignment / finished bundle: do not pursue a conflicting
            # independently proposed frontier while another bundle is active.
            goals[robot_id - 1] = list(positions[robot_id])
        # A pending but paused robot keeps its temporary recovery goal.
    return goals
