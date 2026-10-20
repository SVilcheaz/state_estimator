"""CSV replay adapter shared by the Python estimator implementation."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from estimator_improved import Estimator


EVENT_HEADER = ("timestamp_s", "x_m", "y_m", "z_m")
PREDICTION_HEADER = (
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
OUTPUT_RATE_HZ = 50.0


@dataclass(frozen=True)
class Event:
    timestamp_s: float
    x_m: float
    y_m: float
    z_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    return parser.parse_args()


def load_events(path: Path) -> list[Event]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EVENT_HEADER:
            raise ValueError(f"unexpected event header; expected {','.join(EVENT_HEADER)}")
        events = [
            Event(*(float(row[column]) for column in EVENT_HEADER))
            for row in reader
        ]
    previous_timestamp_s = -math.inf
    for index, event in enumerate(events, start=2):
        if not all(
            math.isfinite(value)
            for value in (event.timestamp_s, event.x_m, event.y_m, event.z_m)
        ):
            raise ValueError(f"non-finite event value at row {index}")
        if event.timestamp_s <= previous_timestamp_s:
            raise ValueError(f"event timestamps must strictly increase at row {index}")
        previous_timestamp_s = event.timestamp_s
    return events


def run(arguments: argparse.Namespace) -> None:
    events = load_events(arguments.input)
    if not events:
        raise ValueError("input must contain at least one measurement")

    estimator = Estimator()
    event_index = 0
    start_time_s = events[0].timestamp_s
    end_time_s = events[-1].timestamp_s
    final_tick = math.floor((end_time_s - start_time_s) * OUTPUT_RATE_HZ + 1e-9)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(PREDICTION_HEADER)
    for tick in range(final_tick + 1):
        query_time_s = start_time_s + tick / OUTPUT_RATE_HZ
        while (
            event_index < len(events)
            and events[event_index].timestamp_s <= query_time_s + 1e-12
        ):
            event = events[event_index]
            estimator.ingest_measurement(
                event.timestamp_s,
                event.x_m,
                event.y_m,
                event.z_m,
            )
            event_index += 1

        state = estimator.predict(query_time_s)
        writer.writerow(
            (
                f"{state.timestamp_s:.9f}",
                f"{state.position_m[0]:.9f}",
                f"{state.position_m[1]:.9f}",
                f"{state.position_m[2]:.9f}",
                f"{state.velocity_mps[0]:.9f}",
                f"{state.velocity_mps[1]:.9f}",
                f"{state.velocity_mps[2]:.9f}",
                "ok",
                1,
            )
        )


def main() -> int:
    try:
        run(parse_args())
        return 0
    except (OSError, ValueError) as error:
        print(f"global_estimator: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
