def find_eye_landmark_csv(session_dir, session_name):
    """
    中文：
    自动寻找 eye landmark CSV。

    这个版本更宽松：
    1. 先找标准位置
    2. 再递归搜索 session 文件夹里的所有 CSV
    3. 再递归搜索 data/processed 里的所有 CSV

    English:
    Automatically locate the eye landmark CSV with broader search.
    """

    possible_paths = [
        PROCESSED_DATA_DIR / f"{session_name}_eye_landmarks.csv",
        session_dir / f"{session_name}_eye_landmarks.csv",
        session_dir / "eye_landmarks.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # 中文：如果标准名字没找到，就在 raw/session 里面递归搜索。
    # English: If standard names fail, recursively search inside the raw session folder.
    session_csvs = list(session_dir.rglob("*.csv"))

    for path in session_csvs:
        name = path.name.lower()

        if "eye" in name or "iris" in name or "landmark" in name:
            return path

    # 中文：再去 processed 文件夹里搜索。
    # English: Also search inside processed folder.
    processed_csvs = list(PROCESSED_DATA_DIR.rglob("*.csv"))

    for path in processed_csvs:
        name = path.name.lower()

        if session_name.lower() in name and ("eye" in name or "iris" in name or "landmark" in name):
            return path

    print("ERROR: Cannot find eye landmark CSV.")
    print("Tried standard locations:")

    for path in possible_paths:
        print(f"  {path}")

    print("\nCSV files found inside this session folder:")
    for path in session_csvs:
        print(f"  {path}")

    print("\nCSV files found inside processed folder:")
    for path in processed_csvs:
        print(f"  {path}")

    return None