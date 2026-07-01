"""Analysis pipeline for the Edinburgh visually guided saccade dataset.

This module deliberately separates measured healthy data from simulated
abnormality. Synthetic rows are useful for testing whether features respond as
expected, but they are not clinical evidence and must not be presented as such.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "SUBID", "BLOCK", "TRIAL", "PHASE", "TOT", "CUR", "ECC", "DX",
    "DY", "AMP", "LAT", "DIFF", "ANSW", "ANTC",
}
PAIR_KEYS = ["SUBID", "BLOCK", "TRIAL", "PHASE"]


@dataclass
class AnalysisSummary:
    experiment: str
    left_rows_loaded: int
    right_rows_loaded: int
    left_primary_rows: int
    right_primary_rows: int
    paired_trials: int
    participants: int
    blocks: int
    healthy_anomaly_threshold: float
    synthetic_rows: int
    synthetic_direction: str
    synthetic_amplitude_scale: float
    synthetic_latency_shift_ms: float
    synthetic_detection_sensitivity: float
    healthy_specificity: float
    balanced_accuracy: float
    caution: str


def load_fixation_sequence(path: Path, eye: str) -> pd.DataFrame:
    """Load a tab-separated fixation sequence and label its source eye."""
    frame = pd.read_csv(path, sep="\t")
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["EYE"] = eye.upper()
    return frame


def select_primary_saccades(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the first valid visually guided response saccade per trial."""
    selected = frame[
        (frame["PHASE"] == 2)
        & (frame["CUR"] == 1)
        & (frame["ANSW"] == 1)
        & (frame["ANTC"] == 0)
        & frame["ECC"].notna()
        & (frame["ECC"].abs() > 0)
        & frame["DX"].notna()
        & frame["LAT"].between(80, 800)
    ].copy()
    selected.sort_values(PAIR_KEYS + ["LAT"], inplace=True)
    return selected.drop_duplicates(PAIR_KEYS, keep="first")


def pair_eyes(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Pair left/right measurements from the same participant and trial."""
    keep = PAIR_KEYS + ["TOT", "ECC", "DX", "DY", "AMP", "LAT", "DIFF"]
    paired = left[keep].merge(
        right[keep], on=PAIR_KEYS, suffixes=("_L", "_R"), validate="one_to_one"
    )
    paired = paired[np.isclose(paired["ECC_L"], paired["ECC_R"], equal_nan=False)].copy()
    paired.rename(columns={"ECC_L": "ECC"}, inplace=True)
    paired.drop(columns=["ECC_R"], inplace=True)
    return compute_features(paired)


def compute_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute interpretable binocular trial-level features."""
    out = frame.copy()
    eccentricity = out["ECC"].abs().replace(0, np.nan)
    out["TARGET_DIRECTION"] = np.where(out["ECC"] >= 0, "right", "left")
    out["GAIN_L"] = out["DX_L"] / out["ECC"]
    out["GAIN_R"] = out["DX_R"] / out["ECC"]
    out["GAIN_MEAN"] = (out["GAIN_L"] + out["GAIN_R"]) / 2
    out["GAIN_ASYMMETRY"] = (out["GAIN_L"] - out["GAIN_R"]).abs()
    out["AMPLITUDE_DIFF_DEG"] = (out["AMP_L"] - out["AMP_R"]).abs()
    out["AMPLITUDE_DIFF_NORM"] = out["AMPLITUDE_DIFF_DEG"] / eccentricity
    out["LATENCY_MEAN_MS"] = (out["LAT_L"] + out["LAT_R"]) / 2
    out["LATENCY_DIFF_MS"] = (out["LAT_L"] - out["LAT_R"]).abs()
    out["VERTICAL_DRIFT_MEAN_DEG"] = (out["DY_L"].abs() + out["DY_R"].abs()) / 2
    out["LANDING_ERROR_PX_MEAN"] = (out["DIFF_L"].abs() + out["DIFF_R"].abs()) / 2
    out["HAS_SECONDARY_SACCADE"] = ((out["TOT_L"] > 1) | (out["TOT_R"] > 1)).astype(int)
    return out


def simulate_directional_restriction(
    healthy: pd.DataFrame,
    eye: str = "R",
    direction: str = "right",
    amplitude_scale: float = 0.55,
    latency_shift_ms: float = 80.0,
) -> pd.DataFrame:
    """Create a clearly labelled, direction-specific synthetic perturbation."""
    if eye not in {"L", "R"}:
        raise ValueError("eye must be 'L' or 'R'")
    if direction not in {"left", "right"}:
        raise ValueError("direction must be 'left' or 'right'")
    if not 0 < amplitude_scale <= 1:
        raise ValueError("amplitude_scale must be in (0, 1]")

    synthetic = healthy[healthy["TARGET_DIRECTION"] == direction].copy()
    synthetic[f"DX_{eye}"] *= amplitude_scale
    synthetic[f"AMP_{eye}"] = np.hypot(synthetic[f"DX_{eye}"], synthetic[f"DY_{eye}"])
    synthetic[f"LAT_{eye}"] += latency_shift_ms
    synthetic = compute_features(synthetic)
    synthetic["LABEL"] = 1
    synthetic["DATA_ORIGIN"] = "synthetic_directional_restriction"
    synthetic["SIMULATED_EYE"] = eye
    synthetic["SIMULATED_DIRECTION"] = direction
    return synthetic


def _robust_location_scale(values: pd.Series, minimum_scale: float) -> tuple[float, float]:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    median = float(finite.median())
    mad = float((finite - median).abs().median())
    q10, q90 = finite.quantile([0.10, 0.90])
    central_scale = float((q90 - q10) / 2.563)
    scale = max(1.4826 * mad, central_scale, minimum_scale)
    return median, scale


def score_anomalies(healthy: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit robust healthy-reference statistics and score a target frame."""
    feature_scales = {
        "GAIN_ASYMMETRY": 0.01,
        "AMPLITUDE_DIFF_NORM": 0.01,
        "LATENCY_DIFF_MS": 5.0,
    }
    reference = {}
    scored = target.copy()
    z_columns = []
    for feature, minimum_scale in feature_scales.items():
        center, scale = _robust_location_scale(healthy[feature], minimum_scale)
        reference[feature] = {"median": center, "robust_scale": scale}
        column = f"ROBUST_Z_{feature}"
        scored[column] = ((scored[feature] - center) / scale).clip(lower=0)
        z_columns.append(column)
    scored["ANOMALY_SCORE"] = scored[z_columns].max(axis=1)
    return scored, reference


def make_subject_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial features into participant-level descriptive statistics."""
    return (
        paired.groupby("SUBID", as_index=False)
        .agg(
            trials=("TRIAL", "size"),
            blocks=("BLOCK", "nunique"),
            median_gain=("GAIN_MEAN", "median"),
            median_latency_ms=("LATENCY_MEAN_MS", "median"),
            median_gain_asymmetry=("GAIN_ASYMMETRY", "median"),
            p95_amplitude_diff_norm=("AMPLITUDE_DIFF_NORM", lambda s: s.quantile(0.95)),
            secondary_saccade_rate=("HAS_SECONDARY_SACCADE", "mean"),
        )
        .sort_values("SUBID")
    )


def _plot_results(healthy: pd.DataFrame, benchmark: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)

    gain = healthy.groupby("ECC", as_index=False).agg(
        gain_left=("GAIN_L", "median"), gain_right=("GAIN_R", "median")
    )
    axes[0, 0].plot(gain["ECC"], gain["gain_left"], marker="o", label="Left eye")
    axes[0, 0].plot(gain["ECC"], gain["gain_right"], marker="o", label="Right eye")
    axes[0, 0].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[0, 0].set(title="Median saccadic gain", xlabel="Target eccentricity (deg)", ylabel="Gain")
    axes[0, 0].legend()

    axes[0, 1].hist(healthy["LATENCY_MEAN_MS"].dropna(), bins=35, color="#4472C4")
    axes[0, 1].set(title="Healthy-reference latency", xlabel="Mean binocular latency (ms)", ylabel="Trials")

    for label, name, color in [(0, "Measured healthy", "#70AD47"), (1, "Synthetic restriction", "#C00000")]:
        subset = benchmark[benchmark["LABEL"] == label]
        axes[1, 0].scatter(
            subset["GAIN_ASYMMETRY"], subset["LATENCY_DIFF_MS"],
            s=12, alpha=0.35, label=name, color=color,
        )
    axes[1, 0].set(title="Measured vs. synthetic features", xlabel="Gain asymmetry", ylabel="Latency difference (ms)")
    axes[1, 0].legend()

    benchmark.boxplot(column="ANOMALY_SCORE", by="DATA_ORIGIN", ax=axes[1, 1], grid=False)
    axes[1, 1].set(title="Robust anomaly score", xlabel="", ylabel="Score")
    fig.suptitle("Edinburgh Saccade Dataset - Exploratory Analysis", fontsize=15, fontweight="bold")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "public_saccade_analysis_dashboard.png", dpi=200)
    plt.close(fig)


def analyze_dataset(
    dataset_dir: Path,
    output_dir: Path,
    experiment: str = "SRE2",
    amplitude_scale: float = 0.55,
    latency_shift_ms: float = 80.0,
) -> AnalysisSummary:
    """Run the full public-data analysis and write reproducible outputs."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    experiment = experiment.upper()
    left = load_fixation_sequence(dataset_dir / f"{experiment}_fixseqL.dat", "L")
    right = load_fixation_sequence(dataset_dir / f"{experiment}_fixseqR.dat", "R")
    left_primary = select_primary_saccades(left)
    right_primary = select_primary_saccades(right)
    paired = pair_eyes(left_primary, right_primary)

    healthy = paired[paired["TARGET_DIRECTION"] == "right"].copy()
    healthy["LABEL"] = 0
    healthy["DATA_ORIGIN"] = "measured_healthy_reference"
    healthy_scored, reference = score_anomalies(healthy, healthy)
    threshold = float(max(3.0, healthy_scored["ANOMALY_SCORE"].quantile(0.99)))
    healthy_scored["PREDICTED_ANOMALY"] = (healthy_scored["ANOMALY_SCORE"] > threshold).astype(int)

    synthetic = simulate_directional_restriction(
        paired, eye="R", direction="right",
        amplitude_scale=amplitude_scale, latency_shift_ms=latency_shift_ms,
    )
    synthetic_scored, _ = score_anomalies(healthy, synthetic)
    synthetic_scored["PREDICTED_ANOMALY"] = (synthetic_scored["ANOMALY_SCORE"] > threshold).astype(int)
    benchmark = pd.concat([healthy_scored, synthetic_scored], ignore_index=True)

    sensitivity = float(synthetic_scored["PREDICTED_ANOMALY"].mean())
    specificity = float(1 - healthy_scored["PREDICTED_ANOMALY"].mean())
    balanced_accuracy = (sensitivity + specificity) / 2

    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "paired_primary_saccades.csv", index=False)
    make_subject_summary(paired).to_csv(output_dir / "subject_summary.csv", index=False)
    benchmark.to_csv(output_dir / "synthetic_benchmark.csv", index=False)
    _plot_results(paired, benchmark, output_dir)

    summary = AnalysisSummary(
        experiment=experiment,
        left_rows_loaded=len(left), right_rows_loaded=len(right),
        left_primary_rows=len(left_primary), right_primary_rows=len(right_primary),
        paired_trials=len(paired), participants=int(paired["SUBID"].nunique()),
        blocks=int(paired["BLOCK"].nunique()), healthy_anomaly_threshold=threshold,
        synthetic_rows=len(synthetic_scored), synthetic_direction="right eye / rightward target",
        synthetic_amplitude_scale=amplitude_scale,
        synthetic_latency_shift_ms=latency_shift_ms,
        synthetic_detection_sensitivity=sensitivity, healthy_specificity=specificity,
        balanced_accuracy=balanced_accuracy,
        caution=("Synthetic detection results validate pipeline behavior only. They do not estimate "
                 "clinical performance and must not be described as diagnosis."),
    )
    payload = asdict(summary)
    payload["robust_reference"] = reference
    payload["dataset_citation"] = (
        "Nuthmann, Vitu, Engbert, and Kliegl (2016), Eye-movement data for visually "
        "guided and memory-guided saccades, DOI: 10.7488/ds/1460, CC BY 4.0."
    )
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
