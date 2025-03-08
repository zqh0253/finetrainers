config=${1:-configs/training/Mtok/mtok_bl_vq.yaml}
ngpus=${2:-8}

export PYTHONPATH=$(dirname $(dirname $(realpath $0))):$PYTHONPATH
export NO_PROXY="${NO_PROXY},mlr-wandb.corp.apple.com"
export no_proxy="${no_proxy},mlr-wandb.corp.apple.com"
export WANDB_BASE_URL="https://mlr-wandb.corp.apple.com"
export WORKSPACE=workspace


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
    conductor s3 cp --recursive s3://jiatao-datasets/misc/mtok/ workspace/models/ 
else
    echo "VGG model already exists at $VGG_PATH, skipping download"
fi

# (optional)
# download previous checkpoint
# mkdir -p ${BOLT_ARTIFACT_DIR}/run3
# conductor s3 cp --recursive s3://bolt-prod-2701045109/tasks/8idmdtvxnw/artifacts/run3/checkpoint-70000 ${BOLT_ARTIFACT_DIR}/run3/checkpoint-70000
# conductor s3 cp --recursive s3://bolt-prod-2701045109/tasks/jxjbb6fqap/artifacts/run3/checkpoint-50000 ${BOLT_ARTIFACT_DIR}/run3/checkpoint-50000
# conductor s3 cp s3://bolt-prod-2701045109/tasks/a3mfww3hva/artifacts/run1/checkpoint-200000/unwrapped_model/pytorch_model.bin workspace/models/

# Check if WANDB_API_KEY is set
if [ -n "${WANDB_API_KEY}" ]; then
    wandb_train_flag="training.enable_wandb=True"
    export WANDB_MODE=online
    echo "WANDB_API_KEY found, enabling wandb logging"
else
    wandb_train_flag="training.enable_wandb=False"
    export WANDB_MODE=offline
    echo "WANDB_API_KEY not found, disabling wandb logging"
fi

# Check experiment name if exists, then override
if [ -n "${EXPERIMENT_NAME}" ]; then
    exp_flag="experiment.name=${EXPERIMENT_NAME}"
    echo "Overriding experiment name to ${EXPERIMENT_NAME}"
fi

# Run training
accelerate launch --config_file $ACC_CONFIG --num_processes=${ngpus} \
scripts/train.py config=$config \
experiment.output_dir=${BOLT_ARTIFACT_DIR}/run3 \
$exp_flag $wandb_train_flag
