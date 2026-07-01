# Gaze Movement Research Prototype

This project explores low-cost eye-movement measurement and reproducible gaze-data analysis. It now contains two complementary workflows:

1. A webcam H-test prototype that records a moving target, extracts iris landmarks with MediaPipe, and produces synchronized traces and overlay videos.
2. A public-data analysis pipeline for the Edinburgh visually guided saccade dataset, including binocular quality features and a clearly labelled synthetic restriction benchmark.

The project is for research and education. It is not a medical device and does not diagnose cranial nerve palsy or any other condition.

## Public dataset

The public-data workflow uses:

Nuthmann, A.; Vitu, F.; Engbert, R.; Kliegl, R. (2016). *Eye-movement data for visually guided and memory-guided saccades*. University of Potsdam. DOI: 10.7488/ds/1460. Licensed under CC BY 4.0.

Expected local location:

```text
D:\download\gaze_public_data\edinburgh_saccades
```

The analysis defaults to the Experiment 2 fixation-sequence files, which contain processed left- and right-eye saccade records from 10 participants. The downloaded Experiment 1 raw EyeLink files are retained for provenance and format inspection.

## Run the public-data analysis

The easiest Windows entry point creates a small Python 3.11 analysis environment on first use:

```powershell
.\run_public_analysis.ps1
```

Manual setup is also available:

```powershell
py -3.11 -m venv .analysis_venv
.\.analysis_venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.analysis_venv\Scripts\python.exe scripts\06_analyze_public_saccades.py
```

Optional arguments can be forwarded through the launcher:

```powershell
.\run_public_analysis.ps1 `
  --dataset-dir D:\download\gaze_public_data\edinburgh_saccades `
  --output-dir D:\download\gaze_public_analysis `
  --experiment SRE2 `
  --amplitude-scale 0.55 `
  --latency-shift-ms 80
```

Generated outputs:

- `paired_primary_saccades.csv`: measured, paired left/right primary saccades.
- `subject_summary.csv`: descriptive statistics by participant.
- `synthetic_benchmark.csv`: measured healthy reference rows plus labelled synthetic perturbations.
- `analysis_summary.json`: sample counts, thresholds, benchmark metrics, and data citation.
- `public_saccade_analysis_dashboard.png`: four-panel exploratory visualization.

## What the synthetic benchmark means

The benchmark reduces right-eye horizontal amplitude and adds latency only for rightward targets. This checks whether binocular asymmetry features react in the expected direction.

It does **not** represent real patient physiology. Its sensitivity and specificity describe separation from a simulated perturbation, not clinical diagnostic performance.

## Webcam workflow

```powershell
python scripts\00_camera_test.py
python scripts\01_collect_h_test.py
python scripts\02_extract_eye_landmarks.py data\raw\session_YYYYMMDD_HHMMSS
python scripts\03_plot_eye_trace.py data\raw\session_YYYYMMDD_HHMMSS
python scripts\05_iris_overlay_video.py data\raw\session_YYYYMMDD_HHMMSS
```

The professional eye-tracker data and webcam data belong to different measurement domains. The public dataset can validate analysis logic and feature behavior, but it cannot by itself establish the accuracy of MediaPipe webcam tracking.
