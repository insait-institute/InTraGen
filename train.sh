export WANDB_KEY=""
export ENTITY="linbin"
export PROJECT="Train"
OUTPUT_DIR_STAGE1="${PROJECT}_stage1"
OUTPUT_DIR_STAGE2="${PROJECT}_stage2"
TRAIN_JSON_PATH="datasets/MoVi-Extended/data_train.json"
VAL_JSON_PATH="datasets/MoVi-Extended/data_val.json"
NUM_OF_GPUS=2

# training stage 1: dense pose
accelerate launch \
    --config_file "scripts/accelerate_configs/train_v2_gpu_$NUM_OF_GPUS.yaml" \
    opensora_stage1/train/train_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "./cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "models/Open-Sora-Plan-v1.1.0/vae" \
    --video_data "$TRAIN_JSON_PATH" \
    --sample_rate 1 \
    --num_frames 65 \
    --max_image_size 512 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=1 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=40000 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --pretrained models/Open-Sora-Plan-v1.1.0/65x512x512/diffusion_pytorch_model.safetensors\
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=8000 \
    --log_validation_steps=2000 \
    --output_dir="$OUTPUT_DIR_STAGE1" \
    --allow_tf32 \
    --model_max_length 300 \
    --use_image_num 0 \
    --enable_tiling \
    --resume_from_checkpoint "latest" \
    --use_img_from_vid \
    --use_deepspeed \
    --validation_json_path "$VAL_JSON_PATH" \
    --enable_tracker


# training stage 2: sparse pose / object id
accelerate launch \
    --config_file scripts/accelerate_configs/train_v2_2gpu.yaml \
    opensora_stage2/train/train_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "./cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "models/Open-Sora-Plan-v1.1.0/vae" \
    --video_data "$TRAIN_JSON_PATH" \
    --sample_rate 1 \
    --num_frames 65 \
    --max_image_size 512 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=1 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=40000 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --pretrained "${OUTPUT_DIR_STAGE1}/checkpoint-32000/model/diffusion_pytorch_model.safetensors"\
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=8000 \
    --log_validation_steps=2000 \
    --output_dir="$OUTPUT_DIR_STAGE2" \
    --allow_tf32 \
    --model_max_length 300 \
    --use_image_num 0 \
    --enable_tiling \
    --resume_from_checkpoint "latest" \
    --use_img_from_vid \
    --use_deepspeed \
    --validation_json_path "$VAL_JSON_PATH" \
    --enable_tracker
