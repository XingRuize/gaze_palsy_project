import cv2


def main():
    """
    中文：测试电脑摄像头能不能正常打开。
    English: Test whether the webcam can be opened successfully.
    """

    # 中文：打开默认摄像头。通常 0 是电脑自带摄像头。
    # English: Open the default webcam. Usually, 0 is the built-in camera.
    cap = cv2.VideoCapture(0)

    # 中文：检查摄像头是否成功打开。
    # English: Check whether the camera was opened successfully.
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        print("中文：如果打不开，可以把 cv2.VideoCapture(0) 改成 cv2.VideoCapture(1)。")
        print("English: If it fails, try changing cv2.VideoCapture(0) to cv2.VideoCapture(1).")
        return

    print("Camera opened successfully.")
    print("中文：按 q 退出。")
    print("English: Press q to quit.")

    while True:
        # 中文：从摄像头读取一帧画面。
        # English: Read one frame from the webcam.
        ret, frame = cap.read()

        # 中文：如果读取失败，就停止。
        # English: If reading fails, stop the loop.
        if not ret:
            print("ERROR: Cannot read frame.")
            break

        # 中文：显示摄像头画面。
        # English: Display the webcam frame.
        cv2.imshow("Camera Test", frame)

        # 中文：等待键盘输入。如果按 q，就退出。
        # English: Wait for keyboard input. If q is pressed, quit.
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # 中文：释放摄像头。
    # English: Release the webcam.
    cap.release()

    # 中文：关闭所有 OpenCV 窗口。
    # English: Close all OpenCV windows.
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()