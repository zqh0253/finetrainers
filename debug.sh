#!/bin/bash
export WANDB_MODE="offline"
#export NCCL_P2P_DISABLE=1
#export TORCH_NCCL_ENABLE_MONITORING=0
#export FINETRAINERS_LOG_LEVEL=DEBUG


DATA_ROOT="video-dataset-disney/videos"
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="debug_ckpt"
ID_TOKEN="BW_STYLE"

# Model arguments
model_cmd="--model_name cogvideox \
  --pretrained_model_name_or_path THUDM/CogVideoX-2b"

# Dataset arguments
dataset_cmd="--data_root $DATA_ROOT \
  --video_column $VIDEO_COLUMN \
  --caption_column $CAPTION_COLUMN \
  --id_token $ID_TOKEN \
  --video_resolution_buckets 21x256x256 \
  --caption_dropout_p 1 \
  --img_dropout_p 0.1 \
  --dataset_type bolt_rgbxyz"
 

# Dataloader arguments
dataloader_cmd="--dataloader_num_workers 6"

# Training arguments
  # --precompute_conditions \
training_cmd="--training_type  full-finetune\
  --seed 42 \
  --batch_size 4 \
  --train_steps 10000 \
  --gradient_accumulation_steps 4 \
  --checkpointing_steps 500 \
  --checkpointing_limit 6 \
  --resume_from_checkpoint=latest \
  --enable_slicing \
  --enable_tiling"

#   --gradient_checkpointing \

# Optimizer arguments
optimizer_cmd="--optimizer adamw \
  --use_8bit_bnb \
  --lr 3e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 100 \
  --lr_num_cycles 1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 1e-5 \
  --epsilon 1e-8 \
  --max_grad_norm 1.0"

# Miscellaneous arguments
miscellaneous_cmd="--tracker_name finetrainers-cog \
  --output_dir $OUTPUT_DIR \
  --nccl_timeout 1800 \
  --report_to wandb"

# cmd="accelerate launch --config_file accelerate_configs/deepspeed.yaml --gpu_ids $GPU_IDS train.py \
cmd="accelerate launch --config_file accelerate_configs/deepspeed_1.yaml --num_processes 8 train.py \
  $model_cmd \
  $dataset_cmd \
  $dataloader_cmd \
  $training_cmd \
  $optimizer_cmd \
  $miscellaneous_cmd"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"
