# 학습 명령어

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset soccer_hybrid \
    --template qwen2_vl \
    --finetuning_type lora \
    --output_dir saves/qwen2vl_soccer_full \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --bf16 True \
    --flash_attn auto \
    --video_fps 1 \
    --video_maxlen 4 \
    --video_max_pixels 100000 \
    --logging_steps 5 \
    --save_steps 100 \
    --plot_loss True
