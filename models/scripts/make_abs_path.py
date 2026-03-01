import json
import os
import argparse

def generate_absolute_paths(base_folder, input_json, output_json):
    """
    상대 경로로 되어 있는 비디오 파일 위치를 실행 환경에 맞는 절대 경로로 변환
    """
    print(f"🔄 경로 변환 시작: {input_json} -> {output_json}")
    
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        modified_count = 0
        for entry in data:
            if "video" in entry:
                if not entry["video"].startswith("/"):
                    abs_path = os.path.abspath(os.path.join(base_folder, entry["video"]))
                    entry["video"] = abs_path
                    modified_count += 1

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ 변환 완료! {modified_count}개의 경로를 '{base_folder}' 기준으로 수정했습니다.")
        
    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert relative video paths to absolute paths for LLaMA-Factory.")
    
    # 사용자가 직접 자기 환경에 맞는 폴더를 입력할 수 있게 인자 설정 
    parser.add_argument("--base_folder", type=str, required=True, 
                        help="비디오 파일(clips/)이 들어있는 최상위 폴더 경로")
    parser.add_argument("--input_json", type=str, default="train_hybrid_ground_truth_final.json",
                        help="원본 데이터셋 파일명")
    parser.add_argument("--output_json", type=str, default="train_absolute_paths.json",
                        help="생성될 절대 경로 데이터셋 파일명")

    args = parser.parse_args()

    generate_absolute_paths(args.base_folder, args.input_json, args.output_json)