import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path


# ============================================================
# Project path settings
# 项目路径设置
# ============================================================

# 中文：当前文件在 scripts 文件夹里，所以 parent.parent 是项目根目录。
# English: This file is inside the scripts folder, so parent.parent is the project root.
BASE_DIR = Path(__file__).resolve().parent.parent

# 中文：处理后数据保存位置。
# English: Location for processed data.
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# 中文：图像结果保存位置。
# English: Location for result plots.
RESULTS_DIR = BASE_DIR / "results"


def resolve_session_dir(path_text):
    """
    中文：
    把用户输入的 session 路径转成绝对路径。
    如果用户输入相对路径，就自动拼到项目根目录下面。

    English:
    Convert a user-provided session path into an absolute path.
    If the user provides a relative path, attach it to the project root.
    """

    session_dir = Path(path_text)

    if not session_dir.is_absolute():
        session_dir = BASE_DIR / session_dir

    return session_dir


def main():
    """
    中文：
    合并红点轨迹和眼睛轨迹，并画出水平/垂直眼动曲线。

    English:
    Merge target trajectory and eye trajectory, then plot horizontal and vertical eye movement traces.
    """

    if len(sys.argv) < 2:
        print("Usage:")
        print("python scripts/03_plot_eye_trace.py data/raw/session_YYYYMMDD_HHMMSS")
        return

    session_dir = resolve_session_dir(sys.argv[1])
    session_name = session_dir.name

    # 中文：红点轨迹文件。
    # English: Target trajectory file.
    target_csv = session_dir / "target_path.csv"

    # 中文：眼睛 landmark 文件。
    # English: Eye landmark file.
    eye_csv = PROCESSED_DATA_DIR / f"{session_name}_eye_landmarks.csv"

    if not target_csv.exists():
        print(f"ERROR: Cannot find target CSV: {target_csv}")
        return

    if not eye_csv.exists():
        print(f"ERROR: Cannot find eye CSV: {eye_csv}")
        return

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    merged_csv = PROCESSED_DATA_DIR / f"{session_name}_merged.csv"

    x_plot = RESULTS_DIR / f"{session_name}_x_trace.png"
    y_plot = RESULTS_DIR / f"{session_name}_y_trace.png"

    print("========== Path Information ==========")
    print(f"Project base directory: {BASE_DIR}")
    print(f"Session folder: {session_dir}")
    print(f"Target CSV: {target_csv}")
    print(f"Eye CSV: {eye_csv}")
    print(f"Merged CSV: {merged_csv}")
    print(f"X plot: {x_plot}")
    print(f"Y plot: {y_plot}")
    print("======================================")

    # 中文：读取两个 CSV。
    # English: Read both CSV files.
    target = pd.read_csv(target_csv)
    eyes = pd.read_csv(eye_csv)

    # 中文：merge_asof 要求按时间排序。
    # English: merge_asof requires both tables to be sorted by time.
    target = target.sort_values("time_sec")
    eyes = eyes.sort_values("time_sec")

    # 中文：
    # 按最近时间点合并眼睛数据和红点数据。
    # 因为视频帧时间和红点记录时间不一定完全一样。
    #
    # English:
    # Merge eye data and target data using the nearest timestamp.
    # Video frame timestamps and target timestamps may not match exactly.
    merged = pd.merge_asof(
        eyes,
        target,
        on="time_sec",
        direction="nearest",
        tolerance=0.05
    )

    # 中文：只保留检测到脸的帧。
    # English: Keep only frames where a face was detected.
    merged = merged[merged["face_detected"] == 1]

    # 中文：保存合并后的数据。
    # English: Save merged data.
    merged.to_csv(merged_csv, index=False)

    # ============================================================
    # Plot horizontal movement
    # 画水平眼动曲线
    # ============================================================

    plt.figure(figsize=(12, 6))

    # 中文：红点水平坐标。
    # English: Horizontal target position.
    plt.plot(
        merged["time_sec"],
        merged["target_x_norm"],
        label="Target X"
    )

    # 中文：左眼虹膜水平坐标。
    # English: Horizontal position of the left iris.
    plt.plot(
        merged["time_sec"],
        merged["left_iris_x_norm"],
        label="Left Iris X"
    )

    # 中文：右眼虹膜水平坐标。
    # English: Horizontal position of the right iris.
    plt.plot(
        merged["time_sec"],
        merged["right_iris_x_norm"],
        label="Right Iris X"
    )

    plt.xlabel("Time seconds")
    plt.ylabel("Normalized horizontal position")
    plt.title("Horizontal Eye Movement vs Target")
    plt.legend()
    plt.grid(True)
    plt.savefig(x_plot, dpi=200)
    plt.close()

    # ============================================================
    # Plot vertical movement
    # 画垂直眼动曲线
    # ============================================================

    plt.figure(figsize=(12, 6))

    # 中文：红点垂直坐标。
    # English: Vertical target position.
    plt.plot(
        merged["time_sec"],
        merged["target_y_norm"],
        label="Target Y"
    )

    # 中文：左眼虹膜垂直坐标。
    # English: Vertical position of the left iris.
    plt.plot(
        merged["time_sec"],
        merged["left_iris_y_norm"],
        label="Left Iris Y"
    )

    # 中文：右眼虹膜垂直坐标。
    # English: Vertical position of the right iris.
    plt.plot(
        merged["time_sec"],
        merged["right_iris_y_norm"],
        label="Right Iris Y"
    )

    plt.xlabel("Time seconds")
    plt.ylabel("Normalized vertical position")
    plt.title("Vertical Eye Movement vs Target")
    plt.legend()
    plt.grid(True)
    plt.savefig(y_plot, dpi=200)
    plt.close()

    print("Done.")
    print(f"Merged CSV saved to: {merged_csv}")
    print(f"X trace saved to: {x_plot}")
    print(f"Y trace saved to: {y_plot}")


if __name__ == "__main__":
    main()