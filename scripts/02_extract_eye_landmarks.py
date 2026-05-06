import cv2
import mediapipe as mp
import pandas as pd
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
# English: Location for saving processed data.
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# MediaPipe iris landmark indices
# MediaPipe 虹膜关键点编号
# ============================================================

# 中文：这些编号代表 MediaPipe Face Mesh 中虹膜附近的 landmark。
# English: These indices represent iris-related landmarks in MediaPipe Face Mesh.
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


def resolve_session_dir(path_text):
    """
    中文：
    把用户输入的 session 路径转成绝对路径。
    如果用户输入的是相对路径，比如 data/raw/session_xxxx，
    就自动拼到项目根目录下面。

    English:
    Convert the user-provided session path into an absolute path.
    If the user provides a relative path such as data/raw/session_xxxx,
    attach it to the project root directory.
    """

    session_dir = Path(path_text)

    if not session_dir.is_absolute():
        session_dir = BASE_DIR / session_dir

    return session_dir


def average_landmarks(landmarks, indices, width, height):
    """
    中文：
    计算几个 landmark 的平均位置，用来近似虹膜中心。

    English:
    Compute the average position of selected landmarks to approximate the iris center.
    """

    xs = []
    ys = []

    for idx in indices:
        lm = landmarks[idx]

        # 中文：lm.x 和 lm.y 是归一化坐标。
        # English: lm.x and lm.y are normalized coordinates.
        xs.append(lm.x)
        ys.append(lm.y)

    x_norm = sum(xs) / len(xs)
    y_norm = sum(ys) / len(ys)

    # 中文：转换成像素坐标。
    # English: Convert to pixel coordinates.
    x_px = x_norm * width
    y_px = y_norm * height

    return x_px, y_px, x_norm, y_norm


def main():
    """
    中文：
    从 webcam.mp4 中提取左右眼虹膜位置。

    English:
    Extract left and right iris positions from webcam.mp4.
    """

    if len(sys.argv) < 2:
        print("Usage:")
        print("python scripts/02_extract_eye_landmarks.py data/raw/session_YYYYMMDD_HHMMSS")
        return

    session_dir = resolve_session_dir(sys.argv[1])
    video_path = session_dir / "webcam.mp4"

    if not video_path.exists():
        print(f"ERROR: Cannot find video: {video_path}")
        return

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    session_name = session_dir.name
    output_csv = PROCESSED_DATA_DIR / f"{session_name}_eye_landmarks.csv"

    print("========== Path Information ==========")
    print(f"Project base directory: {BASE_DIR}")
    print(f"Session folder: {session_dir}")
    print(f"Input video: {video_path}")
    print(f"Output CSV: {output_csv}")
    print("======================================")

    # 中文：打开录制好的视频。
    # English: Open the recorded video.
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Cannot open video.")
        return

    # 中文：读取视频帧率。
    # English: Read video FPS.
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 中文：如果读取失败，就默认使用 30 FPS。
    # English: If FPS reading fails, use 30 FPS as default.
    if fps <= 0:
        fps = 30

    rows = []

    # 中文：初始化 MediaPipe Face Mesh。
    # English: Initialize MediaPipe Face Mesh.
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        frame_id = 0

        while True:
            # 中文：逐帧读取视频。
            # English: Read the video frame by frame.
            ret, frame = cap.read()

            if not ret:
                break

            height, width, _ = frame.shape

            # 中文：用 frame_id / fps 估计当前帧时间。
            # English: Estimate current frame timestamp using frame_id / fps.
            time_sec = frame_id / fps

            # 中文：OpenCV 是 BGR，MediaPipe 需要 RGB。
            # English: OpenCV uses BGR, while MediaPipe expects RGB.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 中文：运行 Face Mesh。
            # English: Run Face Mesh.
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # 中文：计算左眼虹膜中心。
                # English: Compute left iris center.
                left_iris = average_landmarks(
                    landmarks,
                    LEFT_IRIS,
                    width,
                    height
                )

                # 中文：计算右眼虹膜中心。
                # English: Compute right iris center.
                right_iris = average_landmarks(
                    landmarks,
                    RIGHT_IRIS,
                    width,
                    height
                )

                row = {
                    "frame_id": frame_id,
                    "time_sec": time_sec,
                    "face_detected": 1,

                    "left_iris_x_px": left_iris[0],
                    "left_iris_y_px": left_iris[1],
                    "left_iris_x_norm": left_iris[2],
                    "left_iris_y_norm": left_iris[3],

                    "right_iris_x_px": right_iris[0],
                    "right_iris_y_px": right_iris[1],
                    "right_iris_x_norm": right_iris[2],
                    "right_iris_y_norm": right_iris[3],
                }

            else:
                # 中文：如果没有检测到脸，就记录空值。
                # English: If no face is detected, record missing values.
                row = {
                    "frame_id": frame_id,
                    "time_sec": time_sec,
                    "face_detected": 0,

                    "left_iris_x_px": None,
                    "left_iris_y_px": None,
                    "left_iris_x_norm": None,
                    "left_iris_y_norm": None,

                    "right_iris_x_px": None,
                    "right_iris_y_px": None,
                    "right_iris_x_norm": None,
                    "right_iris_y_norm": None,
                }

            rows.append(row)
            frame_id += 1

    cap.release()

    # 中文：保存为 CSV。
    # English: Save as CSV.
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("Done.")
    print(f"Eye landmarks saved to: {output_csv}")


if __name__ == "__main__":
    main()