"""Execute awarded frontier bundles in a fixed map coordinate frame.

Labels belong to the allocation round.  Coordinates are copied when a batch is
installed, so later frontier relabeling cannot redirect an outstanding target.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real


def _cell_coordinate(value):
    if isinstance(value, (str, bytes)):
        raise ValueError("a goal must contain two map-cell coordinates")
    try:
        coordinates = tuple(value)
    except TypeError as error:
        raise ValueError("a goal must contain two map-cell coordinates") from error
    if len(coordinates) != 2:
        raise ValueError("a goal must contain two map-cell coordinates")
    result = []
    for coordinate in coordinates:
        if (isinstance(coordinate, bool) or not isinstance(coordinate, Real)
                or not math.isfinite(coordinate) or coordinate < 0
                or int(coordinate) != coordinate):
            raise ValueError("goal coordinates must be nonnegative integer cells")
        result.append(int(coordinate))
    return tuple(result)


def map_position_xy(planner_pose_inputs, map_resolution, map_shape):
    """Convert planner pose to clipped (column, row) coordinates.

    ``map_resolution`` is in centimetres per cell and ``map_shape`` is
    ``(rows, columns)``.  This helper does not convert persisted goals between
    moving local-map windows; the bundle integration uses a fixed full map.
    """
    if (isinstance(map_resolution, bool) or not isinstance(map_resolution, Real)
            or not math.isfinite(map_resolution) or map_resolution <= 0):
        raise ValueError("map_resolution must be a positive finite number")
    if len(map_shape) != 2 or any(
        isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
        for size in map_shape
    ):
        raise ValueError("map_shape must contain positive row and column counts")
    if len(planner_pose_inputs) != 7:
        raise ValueError("planner_pose_inputs must contain seven values")
    x, y, _, gx1, _, gy1, _ = planner_pose_inputs
    if any(not isinstance(value, Real) or not math.isfinite(value)
           for value in (x, y, gx1, gy1)):
        raise ValueError("planner position and window offsets must be finite")
    column = int(x * 100.0 / map_resolution - gy1)
    row = int(y * 100.0 / map_resolution - gx1)
    return (min(max(column, 0), int(map_shape[1]) - 1),
            min(max(row, 0), int(map_shape[0]) - 1))


@dataclass(frozen=True)
class _Waypoint:
    target: str
    xy: tuple


class BundleExecution:
    """Hold one committed allocation batch until completion or cancellation.

    ``current`` returns ``None`` for an empty or paused queue.  Callers can
    distinguish those cases with ``has_robot_pending``.  A recovery pause does
    not consume a waypoint, and a new auction cannot replace pending work.
    Empty or completed bundles remain managed until cancelled or cleared, so
    their robots wait instead of resuming an unallocated fallback frontier.
    """

    def __init__(self):
        self._queues = {}
        self._paused_until = {}
        self._round_id = None
        self._events = []

    def install(self, bundles, target_coordinates, round_id):
        """Validate the entire batch before changing any execution state."""
        if self.has_pending():
            raise RuntimeError("cannot replace an unfinished bundle batch")
        if not isinstance(bundles, Mapping) or not isinstance(target_coordinates, Mapping):
            raise ValueError("bundles and target_coordinates must be mappings")
        queues = {}
        assigned_targets = set()
        for robot_id, bundle in bundles.items():
            if (isinstance(robot_id, bool) or not isinstance(robot_id, Integral)
                    or robot_id < 1):
                raise ValueError("robot IDs must be positive integers")
            if isinstance(bundle, (str, bytes)):
                raise ValueError("each bundle must be a sequence of target labels")
            try:
                targets = tuple(bundle)
            except TypeError as error:
                raise ValueError("each bundle must be a sequence of target labels") from error
            waypoints = []
            for target in targets:
                if not isinstance(target, str) or not target:
                    raise ValueError("target labels must be nonempty strings")
                if target in assigned_targets:
                    raise ValueError("each target can be awarded only once per batch")
                if target not in target_coordinates:
                    raise ValueError("missing coordinates for target {!r}".format(target))
                assigned_targets.add(target)
                waypoints.append(_Waypoint(target, _cell_coordinate(target_coordinates[target])))
            queues[int(robot_id)] = deque(waypoints)

        self._queues = queues
        self._paused_until = {}
        self._round_id = round_id
        for robot_id, waypoints in queues.items():
            self._events.append({
                "event": "installed", "round_id": round_id, "robot_id": robot_id,
                "targets": [waypoint.target for waypoint in waypoints],
                "goal_coordinates": [waypoint.xy for waypoint in waypoints],
            })

    def current(self, robot_id, step=None):
        queue = self._queues.get(robot_id)
        if not queue or self.is_paused(robot_id, step):
            return None
        return queue[0].xy

    def complete(self, robot_id, goal_xy):
        """Consume exactly the current goal, never a stale or recovery goal."""
        queue = self._queues.get(robot_id)
        if not queue or robot_id in self._paused_until:
            return False
        try:
            goal_xy = _cell_coordinate(goal_xy)
        except ValueError:
            return False
        if goal_xy != queue[0].xy:
            return False
        waypoint = queue.popleft()
        self._events.append({
            "event": "completed", "round_id": self._round_id, "robot_id": robot_id,
            "target": waypoint.target, "goal_xy": waypoint.xy,
            "remaining_count": len(queue),
        })
        return True

    def cancel(self, robot_id, reason):
        queue = self._queues.pop(robot_id, None)
        self._paused_until.pop(robot_id, None)
        if queue is None:
            return False
        if not queue:
            self._events.append({
                "event": "released", "round_id": self._round_id, "robot_id": robot_id,
                "reason": str(reason),
            })
            return True
        self._events.append({
            "event": "cancelled", "round_id": self._round_id, "robot_id": robot_id,
            "reason": str(reason), "targets": [waypoint.target for waypoint in queue],
            "goal_coordinates": [waypoint.xy for waypoint in queue],
        })
        queue.clear()
        return True

    def clear(self, reason):
        for robot_id in list(self._queues):
            self.cancel(robot_id, reason)
        self._queues = {}
        self._paused_until = {}
        self._round_id = None

    def has_pending(self):
        return any(self._queues.values())

    def has_robot_pending(self, robot_id):
        return bool(self._queues.get(robot_id))

    def manages(self, robot_id):
        """Whether this batch owns the robot's execution, including waiting."""
        return robot_id in self._queues

    def pause(self, robot_id, until_step):
        if (isinstance(until_step, bool) or not isinstance(until_step, Integral)
                or until_step < 0):
            raise ValueError("until_step must be a nonnegative integer")
        if not self.has_robot_pending(robot_id):
            return False
        self._paused_until[robot_id] = int(until_step)
        self._events.append({
            "event": "paused", "round_id": self._round_id, "robot_id": robot_id,
            "until_step": int(until_step),
        })
        return True

    def is_paused(self, robot_id, step=None):
        until_step = self._paused_until.get(robot_id)
        if until_step is None:
            return False
        if step is None or step < until_step:
            return True
        del self._paused_until[robot_id]
        self._events.append({
            "event": "resumed", "round_id": self._round_id, "robot_id": robot_id,
            "step": step,
        })
        return False

    def drain_events(self):
        events, self._events = self._events, []
        return events
