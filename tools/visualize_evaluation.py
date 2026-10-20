"""Plot cumulative 3D MAE for all three estimators, with both recordings side by side."""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "tools")]

from baseline import Estimator as Baseline
from estimator import Estimator as CV
from estimator_improved import Estimator as CA
from evaluate import load_recording, query_times, interpolate


def cumulative_mae(estimator, recording):
    measurement_times = [point.time_s for point in recording]
    index = 0
    total_error = 0.0
    times, errors = [], []

    for t in query_times(measurement_times[0], measurement_times[-1]):
        while index < len(recording) and recording[index].time_s <= t + 1e-12:
            point = recording[index]
            estimator.ingest_measurement(point.time_s, point.x_m, point.y_m, point.z_m)
            index += 1
        state = estimator.predict(t)
        if t + 1e-12 < measurement_times[1]:
            continue
        reference = interpolate(recording, measurement_times, t)
        total_error += math.dist(state.position_m, (reference.x_m, reference.y_m, reference.z_m))
        times.append(t - measurement_times[0])
        errors.append(total_error / len(times))
    return times, errors


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, filename in zip(axes, ("target1_xyz.csv", "target2_xyz.csv")):
        path = ROOT / "data" / filename
        recording = load_recording(path)
        for name, estimator, style in (
            ("Baseline", Baseline(), "--"), ("Const-Velo KF", CV(), ":"), ("Const-Accel KF", CA(), "-"),
        ):
            times, errors = cumulative_mae(estimator, recording)
            axis.plot(times, errors, style, label=name)
        title = path.stem.removesuffix("_xyz").replace("target", "Target ")
        axis.set_title(f"Cumulative MAE - {title}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Cumulative 3D MAE (m)")
        axis.set_ylim(bottom=0.0)
        axis.legend(fontsize="small")
    fig.tight_layout()
    output = ROOT / "artifacts" / "evaluation" / "cumulative_mae_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
