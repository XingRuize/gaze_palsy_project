import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime
from pathlib import Path


# ============================================================
# Project path settings
# 项目路径设置
# ============================================================

# 中文：当前文件在 scripts 文件夹里，所以 parent.parent 是项目根目录。
# English: This file is inside the scripts folder, so parent.parent is the project root.
BASE_DIR = Path(__file__).resolve().parent.parent

# 中文：原始数据保存位置。
# English: Location for saving raw data.
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


# ============================================================
# Basic settings
# 基础设置
# ============================================================

# 中文：红点窗口大小。
# English: Size of the red-dot target window.
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700

# 中文：摄像头编号。通常 0 是默认摄像头。如果打不开，改成 1。
# English: Camera index. Usually, 0 is the default webcam. If it fails, try 1.
CAMERA_INDEX = 0

# 中文：保存视频的尺寸和帧率。
# English: Size and frame rate of the saved video.
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
FPS = 30

# 中文：红点半径。
# English: Radius of the red target dot.
POINT_RADIUS = 18

# 中文：红点从一个位置移动到下一个位置所需时间，单位秒。
# English: Time in seconds for the dot to move from one position to another.
MOVE_DURATION = 1.0

# 中文：红点到达某个位置后停留时间，单位秒。
# English: Time in seconds for the dot to stay at each target position.
HOLD_DURATION = 0.5


def make_session_folder():
    """
    中文：
    创建一个新的 session 文件夹，用来保存本次采集的数据。
    例如：
    D:/gaze_palsy_project/data/raw/session_20260506_104019

    English:
    Create a new session folder for this recording.
    Example:
    D:/gaze_palsy_project/data/raw/session_20260506_104019
    """

    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")

    # 中文：强制保存到项目根目录下的 data/raw。
    # English: Force saving under project_root/data/raw.
    session_dir = RAW_DATA_DIR / session_name

    # 中文：自动创建文件夹。如果父文件夹不存在，也会一起创建。
    # English: Create the folder automatically, including parent folders.
    session_dir.mkdir(parents=True, exist_ok=True)

    return session_dir


def interpolate(p1, p2, alpha):
    """
    中文：
    在 p1 和 p2 之间做线性插值，让红点平滑移动。

    English:
    Linearly interpolate between p1 and p2 so the red dot moves smoothly.

    p1: 起点 / starting point, such as (x1, y1)
    p2: 终点 / ending point, such as (x2, y2)
    alpha: 移动进度 / movement progress from 0 to 1
    """

    x = int(p1[0] + (p2[0] - p1[0]) * alpha)
    y = int(p1[1] + (p2[1] - p1[1]) * alpha)

    return x, y


def draw_screen(point):
    """
    中文：创建黑色背景，并在指定位置画红点。
    English: Create a black screen and draw a red dot at the given point.
    """

    # 中文：创建黑色画布。格式是 高度 x 宽度 x 颜色通道。
    # English: Create a black canvas. Format is height x width x color channels.
    screen = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

    # 中文：OpenCV 使用 BGR 颜色格式，(0, 0, 255) 是红色。
    # English: OpenCV uses BGR color format. (0, 0, 255) is red.
    cv2.circle(screen, point, POINT_RADIUS, (0, 0, 255), -1)

    # 中文：显示提示文字。
    # English: Display instruction text.
    cv2.putText(
        screen,
        "Follow the red dot with your eyes. Keep your head still. Press q to quit.",
        (35, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return screen


def build_h_test_path():
    """
    中文：
    定义数字 H-test 的红点移动路径。

    English:
    Define the red-dot movement path for a digital H-test.
    """

    center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

    left = (int(WINDOW_WIDTH * 0.20), WINDOW_HEIGHT // 2)
    right = (int(WINDOW_WIDTH * 0.80), WINDOW_HEIGHT // 2)

    up_left = (int(WINDOW_WIDTH * 0.20), int(WINDOW_HEIGHT * 0.20))
    down_left = (int(WINDOW_WIDTH * 0.20), int(WINDOW_HEIGHT * 0.80))

    up_right = (int(WINDOW_WIDTH * 0.80), int(WINDOW_HEIGHT * 0.20))
    down_right = (int(WINDOW_WIDTH * 0.80), int(WINDOW_HEIGHT * 0.80))

    up_center = (WINDOW_WIDTH // 2, int(WINDOW_HEIGHT * 0.20))
    down_center = (WINDOW_WIDTH // 2, int(WINDOW_HEIGHT * 0.80))

    # 中文：这个路径模拟医生做 H-test 时的移动路线。
    # English: This path simulates the movement route used in a clinical H-test.
    path = [
        center,
        left,
        up_left,
        down_left,
        left,
        center,
        right,
        up_right,
        down_right,
        right,
        center,
        up_center,
        center,
        down_center,
        center
    ]

    return path


def write_frame_and_target(cap, out, writer, start_time, target_point):
    """
    中文：
    读取一帧摄像头画面，写入视频，同时记录当前红点位置。

    English:
    Read one webcam frame, write it to the video, and record the current target position.
    """

    # 中文：从摄像头读取一帧。
    # English: Read one frame from the webcam.
    ret, frame = cap.read()

    if not ret:
        return False

    # 中文：统一视频帧大小。
    # English: Resize the frame to a fixed video size.
    frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

    # 中文：把当前帧写入视频文件。
    # English: Write the current frame to the video file.
    out.write(frame)

    # 中文：计算当前时间戳。
    # English: Compute the current timestamp.
    now = time.perf_counter()
    time_sec = now - start_time

    # 中文：写入红点位置，包括像素坐标和归一化坐标。
    # English: Save the target position in both pixel and normalized coordinates.
    writer.writerow([
        time_sec,
        target_point[0],
        target_point[1],
        target_point[0] / WINDOW_WIDTH,
        target_point[1] / WINDOW_HEIGHT
    ])

    # 中文：显示红点窗口和摄像头预览。
    # English: Show the target window and webcam preview.
    screen = draw_screen(target_point)

    cv2.imshow("H-Test Target", screen)
    cv2.imshow("Camera Preview", frame)

    return True


def main():
    """
    中文：主程序，负责录制一次完整的数字 H-test。
    English: Main program for recording one complete digital H-test.
    """

    # 中文：创建 session 文件夹。
    # English: Create a session folder.
    session_dir = make_session_folder()

    # 中文：定义输出文件路径。
    # English: Define output file paths.
    video_path = session_dir / "webcam.mp4"
    target_csv_path = session_dir / "target_path.csv"

    print("========== Path Information ==========")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project base directory: {BASE_DIR}")
    print(f"Session folder: {session_dir}")
    print(f"Video will be saved to: {video_path}")
    print(f"Target path will be saved to: {target_csv_path}")
    print("======================================")

    # 中文：打开摄像头。
    # English: Open the webcam.
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        print("中文：尝试把 CAMERA_INDEX 从 0 改成 1。")
        print("English: Try changing CAMERA_INDEX from 0 to 1.")
        return

    # 中文：设置摄像头参数。
    # English: Set webcam parameters.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # 中文：创建视频写入器。
    # English: Create the video writer.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    if not out.isOpened():
        print("ERROR: Cannot create video file.")
        print("中文：如果 mp4 保存失败，可以之后改成 avi + XVID。")
        print("English: If mp4 fails, we can later switch to avi + XVID.")
        cap.release()
        return

    # 中文：创建红点路径。
    # English: Create the red-dot path.
    path = build_h_test_path()

    # 中文：创建显示窗口。
    # English: Create display windows.
    cv2.namedWindow("H-Test Target", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("H-Test Target", WINDOW_WIDTH, WINDOW_HEIGHT)

    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Preview", VIDEO_WIDTH, VIDEO_HEIGHT)

    # 中文：记录开始时间。
    # English: Record the start time.
    start_time = time.perf_counter()

    # 中文：打开 CSV 文件，准备写入红点轨迹。
    # English: Open the CSV file for writing target positions.
    with open(target_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 中文：写入 CSV 表头。
        # English: Write the CSV header.
        writer.writerow([
            "time_sec",
            "target_x_px",
            "target_y_px",
            "target_x_norm",
            "target_y_norm"
        ])

        quit_requested = False

        # 中文：逐段移动红点。
        # English: Move the dot segment by segment.
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]

            segment_start = time.perf_counter()

            # 中文：红点从 p1 移动到 p2。
            # English: Move the dot from p1 to p2.
            while True:
                now = time.perf_counter()
                elapsed = now - segment_start
                alpha = min(elapsed / MOVE_DURATION, 1.0)

                target_point = interpolate(p1, p2, alpha)

                ok = write_frame_and_target(
                    cap,
                    out,
                    writer,
                    start_time,
                    target_point
                )

                if not ok:
                    print("ERROR: Cannot read camera frame.")
                    quit_requested = True
                    break

                # 中文：按 q 退出。
                # English: Press q to quit.
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    quit_requested = True
                    break

                if alpha >= 1.0:
                    break

            if quit_requested:
                break

            # 中文：红点到达目标位置后停留一小段时间。
            # English: Hold the dot at the target position for a short time.
            hold_start = time.perf_counter()

            while time.perf_counter() - hold_start < HOLD_DURATION:
                target_point = p2

                ok = write_frame_and_target(
                    cap,
                    out,
                    writer,
                    start_time,
                    target_point
                )

                if not ok:
                    quit_requested = True
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    quit_requested = True
                    break

            if quit_requested:
                break

    # 中文：释放资源。
    # English: Release resources.
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Done.")
    print(f"Video saved to: {video_path}")
    print(f"Target path saved to: {target_csv_path}")


if __name__ == "__main__":
    main()