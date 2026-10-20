from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetState:
    timestamp_s: float
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]


class Estimator:
    """ Implements the constant velocity Kalman Filter."""

    def __init__(self, measurement_error: float = 0.3, process_error: float = 250.0) -> None:
        self._r = measurement_error**2
        self._q = process_error #like motion error
        self.reset()

    def reset(self) -> None:
        self._last_timestamp_s: float | None = None
        self._last_position_m = (0.0, 0.0, 0.0)
        self._velocity_mps = (0.0, 0.0, 0.0)
        self._has_velocity = False
        
        self._pp_var = 0.0
        self._pv_cov = 0.0
        self._vv_var = 0.0

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

        #first measurement to store the first position
        if self._last_timestamp_s is None:
            self._last_timestamp_s = timestamp_s
            self._last_position_m = position_m
            return True

        #second measurement to initialize the velocity and covariance
        if not self._has_velocity:
            elapsed_s = timestamp_s - self._last_timestamp_s
            try:
                velocity_mps = tuple(
                    (current - previous) / elapsed_s
                    for current, previous in zip(position_m, self._last_position_m)
                )
                pp_var = self._r
                pv_cov = self._r / elapsed_s
                vv_var = 2.0 * self._r / elapsed_s**2 + self._q * elapsed_s / 3.0
            except (OverflowError, ZeroDivisionError):
                return False

            if not all(math.isfinite(value) for value in (*velocity_mps, pp_var, pv_cov, vv_var)):
                return False

            self._last_timestamp_s = timestamp_s
            self._last_position_m = position_m
            self._velocity_mps = velocity_mps
            self._pp_var = pp_var
            self._pv_cov = pv_cov
            self._vv_var = vv_var
            self._has_velocity = True
            return True

        # predict step
        elapsed_s = timestamp_s - self._last_timestamp_s
        try:
            #x_ = x + elapsed * velocity
            predicted_position_m = tuple(
                position + velocity * elapsed_s
                for position, velocity in zip(self._last_position_m, self._velocity_mps)
            )
            #dx_ = dx
            predicted_velocity_mps = self._velocity_mps

            #P_ = FPF' + Q
            #F,Q see written notes
            predicted_pp_var = (
                self._pp_var
                + 2.0 * elapsed_s * self._pv_cov
                + elapsed_s**2 * self._vv_var
                + self._q * elapsed_s**3 / 3.0
            )
            predicted_pv_cov = (
                self._pv_cov
                + elapsed_s * self._vv_var
                + self._q * elapsed_s**2 / 2.0
            )
            predicted_vv_var = self._vv_var + self._q * elapsed_s
        except OverflowError:
            return False

        if not all(math.isfinite(value) for value in (
            *predicted_position_m, *predicted_velocity_mps,
            predicted_pp_var, predicted_pv_cov, predicted_vv_var,
        )):
            return False

        # update local prediction
        innovation_var = predicted_pp_var + self._r
        if not math.isfinite(innovation_var) or innovation_var <= 0.0:
            return False

        try:
            #H matrix has only position components
            position_gain = predicted_pp_var / innovation_var
            velocity_gain = predicted_pv_cov / innovation_var

            #v = z - x+
            innovation_m = tuple(
                measured - predicted
                for measured, predicted in zip(position_m, predicted_position_m)
            )

            #x+ = x_ + x_gain*v
            corrected_position_m = tuple(
                predicted + position_gain * innovation
                for predicted, innovation in zip(predicted_position_m, innovation_m)
            )
            #dx+ = dx_ + dx_gain*v
            corrected_velocity_mps = tuple(
                predicted + velocity_gain * innovation
                for predicted, innovation in zip(predicted_velocity_mps, innovation_m)
            )

            #P+ = P_ - WSW'
            pp_var = predicted_pp_var - position_gain**2 * innovation_var
            pv_cov = predicted_pv_cov - position_gain * velocity_gain * innovation_var
            vv_var = predicted_vv_var - velocity_gain**2 * innovation_var
        except OverflowError:
            return False

        if not all(math.isfinite(value) for value in (
            *corrected_position_m, *corrected_velocity_mps, pp_var, pv_cov, vv_var,
        )):
            return False
        if pp_var < 0.0 or vv_var < 0.0:
            return False

        self._last_timestamp_s = timestamp_s
        self._last_position_m = corrected_position_m
        self._velocity_mps = corrected_velocity_mps
        self._pp_var = pp_var
        self._pv_cov = pv_cov
        self._vv_var = vv_var
        return True

    def predict(self, query_time_s: float) -> TargetState:
        if not math.isfinite(query_time_s):
            raise ValueError("query_time_s must be finite")

        if self._last_timestamp_s is None:
            return TargetState(query_time_s, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        if not self._has_velocity:
            return TargetState(query_time_s, self._last_position_m, (0.0, 0.0, 0.0))

        elapsed_s = query_time_s - self._last_timestamp_s
        position_m = tuple(
            position + velocity * elapsed_s
            for position, velocity in zip(self._last_position_m, self._velocity_mps)
        )
        velocity_mps = self._velocity_mps

        if not all(math.isfinite(value) for value in position_m):
            position_m = self._last_position_m
            velocity_mps = (0.0, 0.0, 0.0)

        return TargetState(query_time_s, position_m, velocity_mps)
