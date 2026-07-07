export NCCL_SOCKET_FAMILY=AF_INET
export WANDB_MODE=disabled
export NCCL_IB_DISABLE=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29501
LOCAL_IP=$(ip addr show eth0 | grep -w inet | awk '{print $2}' | cut -d/ -f1)


torchrun \
  --nnodes=8 \
  --nproc_per_node=8 \
  --max_restarts=3 \
  --rdzv_id=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --local_addr=$LOCAL_IP \
  3dgs_train.py