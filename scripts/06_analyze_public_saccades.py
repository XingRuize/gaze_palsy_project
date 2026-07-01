"""Command-line entry point for the public Edinburgh saccade analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from gaze_analysis.public_saccades import analyze_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze paired eyes and a clearly labelled synthetic restriction benchmark."
    )
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path(r"D:\download\gaze_public_data\edinburgh_saccades"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(r"D:\download\gaze_public_analysis"),
    )
    parser.add_argument("--experiment", choices=["SRE1", "SRE2"], default="SRE2")
    parser.add_argument("--amplitude-scale", type=float, default=0.55)
    parser.add_argument("--latency-shift-ms", type=float, default=80.0)
    args = parser.parse_args()

    summary = analyze_dataset(
        args.dataset_dir, args.output_dir, args.experiment,
        args.amplitude_scale, args.latency_shift_ms,
    )
    print("Analysis complete")
    print(f"Measured paired trials: {summary.paired_trials}")
    print(f"Participants: {summary.participants}")
    print(f"Synthetic benchmark sensitivity: {summary.synthetic_detection_sensitivity:.3f}")
    print(f"Healthy specificity: {summary.healthy_specificity:.3f}")
    print(f"Outputs: {args.output_dir}")
    print("IMPORTANT: synthetic benchmark results are not clinical diagnostic performance.")


if __name__ == "__main__":
    main()
