# 데이터 전처리용 코드
# Qwen2-VL 모델과의 호환을 위해, 데이터셋 내 비디오 경로를 리스트 형태로 변환하는 전처리 과정을 수행

import json
import argparse


def convert_video_to_list(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "video" in item and isinstance(item["video"], str):
            item["video"] = [item["video"]]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Video field conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()

    convert_video_to_list(args.file_path)
