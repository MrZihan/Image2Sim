export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=29500
TF_ENABLE_ONEDNN_OPTS=0

CHECKPOINT="checkpoint_pinhole_336x336_fov90"
echo "CHECKPOINT: ${CHECKPOINT}"

torchrun --nproc_per_node=8 --master_port=$MASTER_PORT vln/vln_eval.py --model_path $CHECKPOINT  --habitat_config_path vln/config/vln_r2r.yaml
