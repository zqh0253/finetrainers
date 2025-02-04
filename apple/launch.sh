config=${1:-configs/training/Mtok/mtok_bl_vq.yaml}
export PYTHONPATH=$(dirname $(dirname $(realpath $0))):$PYTHONPATH

# Check and start tensorboard if not running
if ! lsof -i :${TENSORBOARD_PORT} > /dev/null 2>&1; then
    echo "Starting tensorboard on port ${TENSORBOARD_PORT}..."
    tensorboard --port ${TENSORBOARD_PORT} --logdir ${BOLT_ARTIFACT_DIR} --bind_all &
else
    echo "Tensorboard already running on port ${TENSORBOARD_PORT}"
fi

# config accelerate
ACC_CONFIG=${BOLT_ARTIFACT_DIR}/acc_config.yaml
if [ ! -f "$ACC_CONFIG" ]; then
    python apple/generate_accelerate_config.py --config_file $ACC_CONFIG
else
    echo "accelerate config found $ACC_CONFIG"
fi

# download vgg model # HACK: download vgg model is not stable
VGG_PATH="workspace/models/vgg_lpips.pth/vgg.pth"
if [ ! -f "$VGG_PATH" ]; then
    echo "VGG model not found. Downloading..."
    mkdir -p $(dirname "$VGG_PATH")
    conductor s3 cp s3://jiatao-datasets/misc/vgg.pth "$VGG_PATH"
else
    echo "VGG model already exists at $VGG_PATH, skipping download"
fi

# (optional)
# download previous checkpoint
# mkdir -p ${BOLT_ARTIFACT_DIR}/run4
# conductor s3 cp --recursive s3://bolt-prod-2701045109/tasks/a3mfww3hva/artifacts/run1/checkpoint-200000 ${BOLT_ARTIFACT_DIR}/run4/checkpoint-200000
# conductor s3 cp --recursive s3://bolt-prod-2701045109/tasks/pdtxyway6b/artifacts/run1/checkpoint-200000 ${BOLT_ARTIFACT_DIR}/run3/checkpoint-200000
# conductor s3 cp s3://bolt-prod-2701045109/tasks/a3mfww3hva/artifacts/run1/checkpoint-200000/unwrapped_model/pytorch_model.bin workspace/models/

# # launch local training
WORKSPACE=workspace WANDB_MODE=offline \
accelerate launch --config_file $ACC_CONFIG \
scripts/train_mtok.py config=$config \
training.enable_wandb=False \
experiment.output_dir=${BOLT_ARTIFACT_DIR}/run3 \