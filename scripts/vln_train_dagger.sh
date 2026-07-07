export NCCL_SOCKET_FAMILY=AF_INET
export WANDB_MODE=disabled
export NCCL_IB_DISABLE=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29501
LOCAL_IP=$(ip addr show eth0 | grep -w inet | awk '{print $2}' | cut -d/ -f1)
DATASET_CONFIG="scripts/dataset_config.json"
BASE_MODEL="checkpoints/Image2Nav_Qwen3-4B/checkpoint-xxxxx"
MID_RUN_NAME="checkpoints/Image2Nav_Qwen3-4B_Dagger"
SIMULATOR_CKPT="pretrained_models"
PER_DEVICE_BS=1

echo "BASE_MODEL: ${BASE_MODEL}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
echo "DATASET_CONFIG: ${DATASET_CONFIG}"

torchrun \
  --nnodes=8 \
  --nproc_per_node=8 \
  --max_restarts=5 \
  --rdzv_id=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --local_addr=$LOCAL_IP \
  vln/vln_train_dagger.py \
  --deepspeed scripts/zero2.json \
  --model_name_or_path "${BASE_MODEL}" \
  --simulator_ckpt "${SIMULATOR_CKPT}" \
  --dataset_config "${DATASET_CONFIG}" \
  --num_sparse_history_frames 48 \
  --num_recent_observation_frames 16 \
  --num_future_steps 2 \
  --tune_mm_llm True \
  --tune_mm_vision False \
  --tune_mm_mlp True \
  --batch_size ${PER_DEVICE_BS} \
  --bf16 True \
  --run_name "${MID_RUN_NAME}" \
  --output_dir "${MID_RUN_NAME}" \
  --max_steps 10000 \
  --max_train_scenes 100000 \
  --sample_ratios '{"image2sim_batch_1": 2, "image2sim_batch_2": 2, "image2sim_batch_3": 2, "house_grounding": 1, "room_grounding": 1, "r2r": 2, "reverie": 2, "rxr": 4, "srdf": 4}' \
  --per_device_train_batch_size ${PER_DEVICE_BS} \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.075 \
  --lr_scheduler_type "cosine_with_min_lr" \
  --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 65536 \
  --gradient_checkpointing True \
  --dataloader_num_workers 0