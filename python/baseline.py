"""Simple two-point constant-velocity baseline for the coding exercise."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetState:
    timestamp_s: float
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]


class Estimator:
    """Extrapolates velocity from the two latest accepted positions.

    The first measurement is held until a second one defines velocity. This is
    a deliberately basic baseline; keep the public methods when replacing it.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_timestamp_s: float | None = None
        self._last_position_m = (0.0, 0.0, 0.0)
        self._velocity_mps = (0.0, 0.0, 0.0)
        self._has_velocity = False

    def ingest_measurement(
        self,
        timestamp_s: float,
        x_m: float,
        y_m: float,
        z_m: float,
    ) -> bool:
        values = (timestamp_s, x_m, y_m, z_m)
        if not all(math.isfinite(value) for value in values):
            return False
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            return False
        position_m = (x_m, y_m, z_m)
        if self._last_timestamp_s is not None:
            elapsed_s = timestamp_s - self._last_timestamp_s
            velocity_mps = tuple(
                (current - previous) / elapsed_s
                for current, previous in zip(position_m, self._last_position_m)
            )
            self._velocity_mps = (
                velocity_mps
                if all(math.isfinite(value) for value in velocity_mps)
                else (0.0, 0.0, 0.0)
            )
            self._has_velocity = True

        self._last_timestamp_s = timestamp_s
        self._last_position_m = position_m
        return True

    def predict(self, query_time_s: float) -> TargetState:
        if not math.isfinite(query_time_s):
            raise ValueError("query_time_s must be finite")
        if self._last_timestamp_s is None:
            return TargetState(
                query_time_s,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            )
        if not self._has_velocity:
            return TargetState(
                query_time_s,
                self._last_position_m,
                (0.0, 0.0, 0.0),
            )

        elapsed_s = query_time_s - self._last_timestamp_s
        position_m = tuple(
            position + velocity * elapsed_s
            for position, velocity in zip(self._last_position_m, self._velocity_mps)
        )
        velocity_mps = self._velocity_mps
        if not all(math.isfinite(value) for value in position_m):
            position_m = self._last_position_m
            velocity_mps = (0.0, 0.0, 0.0)

        return TargetState(
            query_time_s,
            position_m,
            velocity_mps,
        )
