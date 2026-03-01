# 1. 절대 경로로 변환
# --base_folder에 비디오 파일(clips/)이 실제로 들어있는 경로를 입력하세요.
# 실행 전 이 부분을 본인의 환경에 맞게 수정해야 합니다.
python scripts/make_abs_path.py \
    --base_folder "/home/user/my_project/data" \
    --input_json "train_hybrid_ground_truth_final.json" \
    --output_json "train_absolute_paths.json"

# 2. 리스트 형식으로 최종 전처리 (Qwen2-VL 호환성 작업)
python scripts/prepare_dataset.py --file_path train_absolute_paths.json

echo "✅ 모든 데이터 준비가 완료되었습니다!"
echo "🚀 이제 LLaMA-Factory를 통해 학습을 시작할 수 있습니다."