from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetState:
    timestamp_s: float
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]


def _transpose(matrix):
    return tuple(zip(*matrix))


def _multiply(left, right):
    columns = _transpose(right)
    return tuple(tuple(sum(a * b for a, b in zip(row, column))
                       for column in columns) for row in left)


class Estimator:
    """Implements the constant acceleration Kalman Filter."""

    def __init__(
        self, measurement_error: float = 0.3, process_error: float = 30.0,
        initial_acceleration_std: float = 10.0,
    ) -> None:
        self._r = measurement_error ** 2
        self._q = process_error
        self._initial_acceleration_var = initial_acceleration_std ** 2
        self.reset()

    def reset(self) -> None:
        self._last_timestamp_s: float | None = None
        self._position_m = (0.0, 0.0, 0.0)
        self._velocity_mps = (0.0, 0.0, 0.0)
        self._acceleration_mps2 = (0.0, 0.0, 0.0)
        self._covariance = ((0.0, 0.0, 0.0),) * 3
        self._has_velocity = False

    def ingest_measurement(
        self, timestamp_s: float, x_m: float, y_m: float, z_m: float,
    ) -> bool:
        if not all(math.isfinite(v) for v in (timestamp_s, x_m, y_m, z_m)):
            return False
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            return False
        measurement = (x_m, y_m, z_m)

        #store first measurement
        if self._last_timestamp_s is None:
            self._last_timestamp_s = timestamp_s
            self._position_m = measurement
            return True

        dt = timestamp_s - self._last_timestamp_s
        try:
            if not self._has_velocity:
                position = measurement
                velocity = tuple((z - p) / dt for z, p in zip(measurement, self._position_m))
                acceleration = (0.0, 0.0, 0.0)

                a_var = self._initial_acceleration_var
                pp = self._r
                pv = self._r / dt
                vv = 2.0 * self._r / dt**2 + a_var * dt**2 / 4.0 + 2.0 * self._q * dt**3 / 15.0
                va = a_var * dt / 2.0 + self._q * dt**2 / 3.0
                aa = a_var + self._q * dt
                covariance = ((pp, pv, 0.0), (pv, vv, va), (0.0, va, aa))
            else:
                #white jerk: p'=v, v'=a, a'=white noise.
                half_dt2 = 0.5 * dt**2
                predicted_position = tuple(
                    p + v * dt + a * half_dt2
                    for p, v, a in zip(self._position_m, self._velocity_mps, self._acceleration_mps2)
                )
                predicted_velocity = tuple(
                    v + a * dt for v, a in zip(self._velocity_mps, self._acceleration_mps2)
                )
                transition = ((1.0, dt, half_dt2), (0.0, 1.0, dt), (0.0, 0.0, 1.0))
                process = (
                    (dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0),
                    (dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0),
                    (dt**3 / 6.0, dt**2 / 2.0, dt),
                )
                propagated = _multiply(_multiply(transition, self._covariance), _transpose(transition))
                predicted_cov = tuple(tuple(propagated[i][j] + self._q * process[i][j]
                                            for j in range(3)) for i in range(3))

                innovation_var = predicted_cov[0][0] + self._r
                if not math.isfinite(innovation_var) or innovation_var <= 0.0:
                    return False
                gain = tuple(predicted_cov[i][0] / innovation_var for i in range(3))
                innovation = tuple(z - p for z, p in zip(measurement, predicted_position))
                position = tuple(p + gain[0] * e for p, e in zip(predicted_position, innovation))
                velocity = tuple(v + gain[1] * e for v, e in zip(predicted_velocity, innovation))
                acceleration = tuple(a + gain[2] * e for a, e in zip(self._acceleration_mps2, innovation))

                corrected = tuple(tuple(predicted_cov[i][j] - gain[i] * innovation_var * gain[j]
                                        for j in range(3)) for i in range(3))
                covariance = tuple(tuple(0.5 * corrected[i][j] + 0.5 * corrected[j][i]
                                         for j in range(3)) for i in range(3))
        except (OverflowError, ZeroDivisionError):
            return False

        values = (*position, *velocity, *acceleration, *(v for row in covariance for v in row))
        if not all(math.isfinite(v) for v in values):
            return False
        if any(covariance[i][i] < 0.0 for i in range(3)):
            return False
        self._last_timestamp_s = timestamp_s
        self._position_m = position
        self._velocity_mps = velocity
        self._acceleration_mps2 = acceleration
        self._covariance = covariance
        self._has_velocity = True
        return True

    def predict(self, query_time_s: float) -> TargetState:
        if not math.isfinite(query_time_s):
            raise ValueError("query_time_s must be finite")
        
        if self._last_timestamp_s is None or not self._has_velocity:
            return TargetState(query_time_s, self._position_m, (0.0, 0.0, 0.0))
        
        dt = query_time_s - self._last_timestamp_s
        try:
            position = tuple(p + v * dt + 0.5 * a * dt**2 for p, v, a in zip(
                self._position_m, self._velocity_mps, self._acceleration_mps2
            ))
            velocity = tuple(v + a * dt for v, a in zip(self._velocity_mps, self._acceleration_mps2))
        except OverflowError:
            return TargetState(query_time_s, self._position_m, (0.0, 0.0, 0.0))
        
        if not all(math.isfinite(v) for v in (*position, *velocity)):
            return TargetState(query_time_s, self._position_m, (0.0, 0.0, 0.0))
        return TargetState(query_time_s, position, velocity)
