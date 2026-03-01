# Google Colab 환경에서 학습 속도를 높이기 위해 데이터를 로컬 SSD로 옮기고 경로를 업데이트
# 구글 드라이브의 영상을 코랩 로컬 SSD(/content)로 복사하고 JSON 내의 비디오 경로를 로컬로 업데이트하여 학습 속도를 최적화합니다.

import os
import json
import shutil
import argparse

def setup_colab_fast_io(drive_video_path, drive_json_path):
    local_video_dir = "/content/clips"
    local_json_path = "/content/train_local.json"
    
    if not os.path.exists(local_video_dir):
        print(f"🚀 영상 데이터를 로컬 SSD로 복사 중: {drive_video_path} -> {local_video_dir}")
        shutil.copytree(drive_video_path, local_video_dir)
        print("✅ 영상 복사 완료!")
    else:
        print("ℹ️ 이미 로컬에 영상 데이터가 존재합니다.")

    print(f"📝 JSON 경로 업데이트 중: {drive_json_path}")
    with open(drive_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified_count = 0
    for entry in data:
        if "video" in entry:
            orig_path = entry["video"][0] if isinstance(entry["video"], list) else entry["video"]
            filename = os.path.basename(orig_path)
            
            new_path = os.path.join(local_video_dir, filename)
            entry["video"] = [new_path] if isinstance(entry["video"], list) else new_path
            modified_count += 1

    with open(local_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ 설정 완료! 새 JSON 저장 위치: {local_json_path}")
    print(f"📊 총 {modified_count}개의 데이터 경로가 로컬 SSD 기준으로 변경되었습니다.")
    
    return local_json_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Training Speed on Google Colab")
    parser.add_argument("--drive_videos", type=str, required=True, help="구글 드라이브 내 영상 폴더 경로")
    parser.add_argument("--drive_json", type=str, required=True, help="구글 드라이브 내 원본 JSON 파일 경로")

    args = parser.parse_args()
    setup_colab_fast_io(args.drive_videos, args.drive_json)