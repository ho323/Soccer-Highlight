# LLaMA-Factory 명령어를 사용하여 모델 병합 및 저장 (학습 끝난 후 실행)

llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --adapter_name_or_path saves/qwen2vl_soccer_full \
    --template qwen2_vl \
    --export_dir saves/qwen2vl_soccer_merged \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False
