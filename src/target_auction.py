"""Exact target-bundle allocation and Clarke-pivot settlement.

The paper's common valuation scale applies to *all* nonempty bundles.  With
zero reserve and no capacity change, a positive common scale leaves the winning
allocation unchanged while scaling valuations, payments, and net utilities.

``reserve_price`` is the seller's value for retaining each unallocated target.
It enters the same welfare objective in the actual and counterfactual solves;
it is not a lower bound applied to an already computed payment.  Settlement
does not reorder winners by their utility or introduce a robot budget.
"""

from dataclasses import dataclass, replace
from math import fsum, isclose, isfinite
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, Optional


def _nonnegative_number(value, name):
    if isinstance(value, bool):
        raise ValueError("{} must be a finite nonnegative number".format(name))
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} must be a finite nonnegative number".format(name)) from exc
    if not isfinite(number) or number < 0:
        raise ValueError("{} must be a finite nonnegative number".format(name))
    return number


def _canonical_ids(values, name):
    ids = tuple(values)
    try:
        unique = set(ids)
    except TypeError as exc:
        raise ValueError("{} must contain hashable identifiers".format(name)) from exc
    if len(unique) != len(ids):
        raise ValueError("{} must not contain duplicates".format(name))
    # Numeric/string identifiers are used by the navigation code.  Sorting by
    # type and representation also gives deterministic order for mixed types.
    return tuple(sorted(ids, key=lambda value: (type(value).__name__, repr(value))))


@dataclass(frozen=True)
class AuctionResult:
    """A complete allocation and settlement, including robots with no bundle."""

    bundles: Mapping
    valuations: Mapping
    payments: Mapping
    net_utilities: Mapping
    counterfactual_welfare: Mapping
    welfare: float
    seller_value: float
    unsold_targets: tuple
    settled: bool = False

    def as_dict(self):
        """Return ordinary containers for structured experiment logs."""
        return {
            "bundles": {robot: list(bundle) for robot, bundle in self.bundles.items()},
            "valuations": dict(self.valuations),
            "payments": dict(self.payments),
            "net_utilities": dict(self.net_utilities),
            "counterfactual_welfare": dict(self.counterfactual_welfare),
            "welfare": self.welfare,
            "seller_value": self.seller_value,
            "unsold_targets": list(self.unsold_targets),
            "settled": self.settled,
        }


@dataclass(frozen=True)
class _Allocation:
    bundles: Mapping
    welfare: float
    seller_value: float
    unsold_targets: tuple


class PriorityAuctionSystem:
    """Solve the small frontier WDP exactly and settle all robots together.

    Each robot receives at most one (possibly empty) bundle, and bundles are
    disjoint.  ``max_targets_per_robot=None`` allows an arbitrary bundle; an
    integer imposes its maximum size.  Missing robot-target valuations are
    unavailable edges, so a robot never wins a target on which it did not bid.

    The subset dynamic program has O(robots * 3**targets) transitions.  The
    navigation system exposes at most four frontier targets.  Exact numerical
    ties prefer fewer allocated targets (avoid zero-surplus allocation), then
    the canonical robot order for each target.  Input dictionary order has no
    effect on this tie rule.
    """

    def __init__(self, agents, targets, r3_inputs, valuation_scale=0.8,
                 max_targets_per_robot=None, reserve_price=0.0):
        self.agents = _canonical_ids(agents, "agents")
        self.targets = _canonical_ids(targets, "targets")
        self.valuation_scale = _nonnegative_number(valuation_scale, "valuation_scale")
        self.reserve_price = _nonnegative_number(reserve_price, "reserve_price")
        if max_targets_per_robot is not None:
            if (isinstance(max_targets_per_robot, bool)
                    or not isinstance(max_targets_per_robot, Integral)
                    or max_targets_per_robot < 0):
                raise ValueError("max_targets_per_robot must be a nonnegative integer or None")
            max_targets_per_robot = int(max_targets_per_robot)
        self.max_targets_per_robot = max_targets_per_robot
        if not isinstance(r3_inputs, Mapping):
            raise ValueError("r3_inputs must be a mapping of robot valuations")
        if set(r3_inputs) - set(self.agents):
            raise ValueError("r3_inputs contains an unknown robot")
        self.r3_inputs = {}
        for robot in self.agents:
            row = r3_inputs.get(robot, {})
            if not isinstance(row, Mapping):
                raise ValueError("each robot's valuations must be a mapping")
            if set(row) - set(self.targets):
                raise ValueError("r3_inputs contains an unknown target")
            self.r3_inputs[robot] = {
                target: _nonnegative_number(value, "valuation")
                for target, value in row.items()
            }
        self.bundle_valuations = {}
        self._choices = {}
        self._precompute_bundle_values()
        self._reset_settlement()

    @property
    def RESERVE_PRICE(self):
        """Compatibility name for the configured per-target seller value."""
        return self.reserve_price

    def _reset_settlement(self):
        self.assignments = {}
        self.agent_assignments = {}
        self.payments = {}
        self.net_utilities = {}
        self.result: Optional[AuctionResult] = None

    def _precompute_bundle_values(self):
        limit = (len(self.targets) if self.max_targets_per_robot is None
                 else self.max_targets_per_robot)
        for robot in self.agents:
            choices = [(0, 0.0)]
            row = self.r3_inputs[robot]
            for mask in range(1, 1 << len(self.targets)):
                bundle = tuple(target for index, target in enumerate(self.targets)
                               if mask & (1 << index))
                if len(bundle) > limit or any(target not in row for target in bundle):
                    continue
                value = self.valuation_scale * fsum(row[target] for target in bundle)
                if not isfinite(value):
                    raise ValueError("scaled bundle valuation must be finite")
                self.bundle_valuations[(robot, bundle)] = value
                choices.append((mask, value))
            self._choices[robot] = tuple(choices)

    def _owner_key(self, masks):
        return tuple(next((index for index, mask in enumerate(masks)
                           if mask & (1 << target)), len(self.agents))
                     for target in range(len(self.targets)))

    def _solve(self, excluded_agents=None):
        excluded = set(() if excluded_agents is None else excluded_agents)
        if excluded - set(self.agents):
            raise ValueError("excluded_agents contains an unknown robot")
        available = tuple(robot for robot in self.agents if robot not in excluded)
        # used mask -> (sum of robot valuations, per-robot chosen masks)
        states = {0: (0.0, ())}
        for robot in available:
            next_states = {}
            for used, (value, masks) in states.items():
                for mask, bundle_value in self._choices[robot]:
                    if used & mask:
                        continue
                    new_used = used | mask
                    candidate = (value + bundle_value, masks + (mask,))
                    incumbent = next_states.get(new_used)
                    if (incumbent is None or candidate[0] > incumbent[0]
                            or (candidate[0] == incumbent[0]
                                and self._owner_key(candidate[1]) < self._owner_key(incumbent[1]))):
                        next_states[new_used] = candidate
            states = next_states
        best = None
        for used, (value, masks) in states.items():
            count = bin(used).count("1")
            seller_value = self.reserve_price * (len(self.targets) - count)
            welfare = value + seller_value
            if not isfinite(welfare):
                raise ValueError("total welfare must be finite")
            key = (count, self._owner_key(masks))
            if best is None or welfare > best[0] or (welfare == best[0] and key < best[1]):
                best = (welfare, key, used, masks, seller_value)
        welfare, _, used, masks, seller_value = best
        bundles = {robot: () for robot in self.agents}
        for robot, mask in zip(available, masks):
            bundles[robot] = tuple(target for index, target in enumerate(self.targets)
                                   if mask & (1 << index))
        unsold = tuple(target for index, target in enumerate(self.targets)
                       if not used & (1 << index))
        return _Allocation(bundles, welfare, seller_value, unsold)

    def solve_winner_determination(self, excluded_agents=None):
        """Return optimal welfare, including the seller's retained-target value."""
        return self._solve(excluded_agents).welfare

    def validate_settlement(self, result=None):
        """Validate the allocation and Clarke-pivot identities before execution."""
        result = self.result if result is None else result
        if result is None:
            raise ValueError("no auction settlement is available")
        expected_robots = set(self.agents)
        for field in (result.bundles, result.valuations, result.payments,
                      result.net_utilities, result.counterfactual_welfare):
            if set(field) != expected_robots:
                raise ValueError("settlement does not include every robot")
        owned = set()
        for robot in self.agents:
            bundle = result.bundles[robot]
            if (len(set(bundle)) != len(bundle) or owned.intersection(bundle)
                    or set(bundle) - set(self.r3_inputs[robot])):
                raise ValueError("settlement has duplicate, conflicting, or unavailable targets")
            if (self.max_targets_per_robot is not None
                    and len(bundle) > self.max_targets_per_robot):
                raise ValueError("settlement exceeds a robot's target capacity")
            owned.update(bundle)
            valuation = self.bundle_valuations.get((robot, tuple(bundle)), 0.0)
            payment = result.payments[robot]
            utility = result.net_utilities[robot]
            expected_payment = result.counterfactual_welfare[robot] - (result.welfare - valuation)
            values = (valuation, payment, utility, result.counterfactual_welfare[robot])
            if any(not isfinite(value) for value in values):
                raise ValueError("settlement contains a nonfinite value")
            close = lambda left, right: isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)
            if (not close(result.valuations[robot], valuation)
                    or not close(payment, expected_payment)
                    or not close(utility, valuation - payment)
                    or payment < -1e-10 or payment > valuation + 1e-10):
                raise ValueError("invalid VCG payment or net-utility settlement")
        unsold = tuple(target for target in self.targets if target not in owned)
        if result.unsold_targets != unsold:
            raise ValueError("settlement has inconsistent unsold targets")
        seller_value = self.reserve_price * len(unsold)
        if (not isclose(result.seller_value, seller_value, rel_tol=1e-10, abs_tol=1e-10)
                or not isclose(result.welfare, fsum(result.valuations.values()) + seller_value,
                               rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError("settlement has inconsistent welfare")
        return True

    def auction_process(self):
        """Settle the complete exact allocation; return legacy immediate targets.

        Callers that execute bundles must read ``get_settled_result().bundles``.
        The legacy return maps winners to their first target for old entrypoints.
        """
        self._reset_settlement()
        allocation = self._solve()
        valuations = {
            robot: self.bundle_valuations.get((robot, allocation.bundles[robot]), 0.0)
            for robot in self.agents
        }
        counterfactual = {robot: self._solve([robot]).welfare for robot in self.agents}
        payments = {}
        for robot in self.agents:
            payment = counterfactual[robot] - (allocation.welfare - valuations[robot])
            # Remove only floating-point cancellation at zero; no reserve floor.
            if payment < 0.0 and isclose(payment, 0.0, abs_tol=1e-10):
                payment = 0.0
            payments[robot] = payment
        utilities = {robot: valuations[robot] - payments[robot] for robot in self.agents}
        result = AuctionResult(
            bundles=MappingProxyType(dict(allocation.bundles)),
            valuations=MappingProxyType(valuations),
            payments=MappingProxyType(payments),
            net_utilities=MappingProxyType(utilities),
            counterfactual_welfare=MappingProxyType(counterfactual),
            welfare=allocation.welfare,
            seller_value=allocation.seller_value,
            unsold_targets=allocation.unsold_targets,
        )
        self.validate_settlement(result)
        self.result = replace(result, settled=True)
        self.assignments = {robot: bundle for robot, bundle in self.result.bundles.items() if bundle}
        self.agent_assignments = {
            robot: bundle[0] if len(bundle) == 1 else bundle
            for robot, bundle in self.assignments.items()
        }
        self.payments = dict(self.result.payments)
        self.net_utilities = dict(self.result.net_utilities)
        return {robot: bundle[0] for robot, bundle in self.assignments.items()}

    def get_settled_result(self):
        """Return the validated result used to authorize a navigation commit."""
        if self.result is None or not self.result.settled:
            raise ValueError("auction must be settled before its targets can be executed")
        self.validate_settlement()
        return self.result

    def calculate_vcg_payment(self, winner, won_bundle):
        """Compatibility access to the payment for the completed allocation."""
        if self.result is None:
            self.auction_process()
        result = self.get_settled_result()
        if winner not in result.bundles:
            raise ValueError("unknown winning robot")
        bundle = (won_bundle,) if isinstance(won_bundle, str) else tuple(won_bundle)
        if bundle != result.bundles[winner]:
            raise ValueError("payment must refer to the robot's actual settled bundle")
        return result.payments[winner]
