#!/usr/bin/env python3
"""Print the MAE of one estimator on one target recording."""

from __future__ import annotations

import argparse
import bisect
import csv
import io
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


RATE_HZ = 50.0
TIMEOUT_S = 30.0
OUTPUT_HEADER = (
    "timestamp_s",
    "x_m",
    "y_m",
    "z_m",
    "vx_mps",
    "vy_mps",
    "vz_mps",
    "status",
    "valid",
)


@dataclass(frozen=True)
class Point:
    time_s: float
    x_m: float
    y_m: float
    z_m: float


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    return parser.parse_args()


def finite(value: str | None, description: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {description}: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {description}: {value!r}")
    return number


def load_recording(path: Path) -> list[Point]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ("timestamp_s", "x_m", "y_m", "z_m"):
            raise ValueError(f"Unexpected recording header in {path}")
        points = [
            Point(
                finite(row["timestamp_s"], "recording timestamp"),
                finite(row["x_m"], "recording x"),
                finite(row["y_m"], "recording y"),
                finite(row["z_m"], "recording z"),
            )
            for row in reader
        ]

    if len(points) < 2:
        raise ValueError(f"Recording must contain at least two points: {path}")
    if any(right.time_s <= left.time_s for left, right in zip(points, points[1:])):
        raise ValueError(f"Recording timestamps must increase strictly: {path}")
    return points


def interpolate(recording: Sequence[Point], times: Sequence[float], time_s: float) -> Point:
    if time_s <= recording[0].time_s:
        point = recording[0]
        return Point(time_s, point.x_m, point.y_m, point.z_m)
    if time_s >= recording[-1].time_s:
        point = recording[-1]
        return Point(time_s, point.x_m, point.y_m, point.z_m)

    right_index = bisect.bisect_right(times, time_s)
    left = recording[right_index - 1]
    right = recording[right_index]
    fraction = (time_s - left.time_s) / (right.time_s - left.time_s)
    return Point(
        time_s,
        left.x_m + fraction * (right.x_m - left.x_m),
        left.y_m + fraction * (right.y_m - left.y_m),
        left.z_m + fraction * (right.z_m - left.z_m),
    )


def query_times(start_time_s: float, end_time_s: float) -> list[float]:
    final_tick = math.floor((end_time_s - start_time_s) * RATE_HZ + 1e-9)
    return [start_time_s + tick / RATE_HZ for tick in range(final_tick + 1)]


def run_estimator(estimator: Path, scenario: Path) -> str:
    completed = subprocess.run(
        (str(estimator), "--input", str(scenario)),
        cwd=scenario.parent,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0:
        details = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or "no process output"
        )
        raise RuntimeError(f"Estimator exited with {completed.returncode}: {details}")
    return completed.stdout


def score_predictions(
    output: str,
    expected_times: Sequence[float],
    truth: Sequence[Point],
    score_start_s: float,
) -> float:
    reader = csv.DictReader(io.StringIO(output))
    if tuple(reader.fieldnames or ()) != OUTPUT_HEADER:
        raise ValueError("Unexpected estimator output")
    rows = list(reader)
    if len(rows) != len(expected_times):
        raise ValueError(f"Expected {len(expected_times)} predictions, received {len(rows)}")

    errors: list[float] = []
    for row_number, (row, expected_time_s, target) in enumerate(
        zip(rows, expected_times, truth), start=2
    ):
        values = [
            finite(row[column], f"{column} at row {row_number}")
            for column in OUTPUT_HEADER[:7]
        ]
        if abs(values[0] - expected_time_s) > 1e-6:
            raise ValueError(f"Wrong timestamp at prediction row {row_number}")
        status = (row["status"] or "").strip().lower()
        if status != "ok":
            raise ValueError(f"Prediction status is not ok at row {row_number}")
        valid = (row["valid"] or "").strip().lower()
        if valid not in {"1", "true"}:
            raise ValueError(f"Prediction is not valid at row {row_number}")

        if expected_time_s + 1e-12 < score_start_s:
            continue
        errors.append(
            math.sqrt(
                (values[1] - target.x_m) ** 2
                + (values[2] - target.y_m) ** 2
                + (values[3] - target.z_m) ** 2
            )
        )

    if not errors:
        raise ValueError("No predictions available after the second measurement")
    return statistics.fmean(errors)


def evaluate(estimator: Path, scenario: Path) -> float:
    estimator = estimator.expanduser().resolve()
    scenario = scenario.expanduser().resolve()
    if not estimator.is_file():
        raise FileNotFoundError(f"Estimator executable does not exist: {estimator}")
    if not scenario.is_file():
        raise FileNotFoundError(f"Scenario recording does not exist: {scenario}")

    recording = load_recording(scenario)
    recording_times = [point.time_s for point in recording]
    expected_times = query_times(recording[0].time_s, recording[-1].time_s)
    truth = [interpolate(recording, recording_times, time_s) for time_s in expected_times]
    output = run_estimator(estimator, scenario)
    return score_predictions(output, expected_times, truth, recording[1].time_s)


def main() -> int:
    try:
        options = arguments()
        print(f"MAE: {evaluate(options.estimator, options.scenario):.3f} m")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
