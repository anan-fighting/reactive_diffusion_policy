#!/bin/bash
# ================================================================
# 用 Dobot 自定义数据集训练 RDP（AT + LDP）
# 使用前请先运行数据转换脚本：
#   python extract_dobot_tactile_markers_and_pca.py   # 生成触觉相关的数据
#   python scripts/convert_dobot_to_zarr.py           # 将原始数据转换为 zarr 格式
# ================================================================

GPU_ID=0

DATASET_PATH="data/hf_dataset/dataset_mini/dobot_peg_in_hole_zarr"
LOGGING_MODE="disabled"   # "online" (wandb) 或 "disabled"

TIMESTAMP=$(date +%m%d%H%M%S)
SEARCH_PATH="./data/outputs"

# ── Stage 1: 训练 Asymmetric Tokenizer ──────────────────────────
echo "Stage 1: Training Asymmetric Tokenizer (AT)..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --config-name=train_at_workspace \
    task=dobot_rdp_image_tactile_emb_at_24fps \
    task.dataset_path=${DATASET_PATH} \
    task.name=dobot_rdp_image_tactile_emb_at_24fps_${TIMESTAMP} \
    at=at_dobot_rdp \
    logging.mode=${LOGGING_MODE}

# ── 查找 loss 最小的 AT checkpoint（topk ckpt，非 latest.ckpt）────
echo ""
echo "Searching for the best AT checkpoint (lowest train_loss)..."
AT_CKPT_DIR=$(find "${SEARCH_PATH}" -maxdepth 2 -path "*${TIMESTAMP}*" -type d)/checkpoints

# 在 checkpoints 目录下找 epoch=XXXX-train_loss=XXXX.ckpt，取 loss 最小的那个
AT_LOAD_DIR=$(ls "${AT_CKPT_DIR}"/epoch=*.ckpt 2>/dev/null \
    | awk -F'train_loss=' '{print $2, $0}' \
    | sort -n | head -1 | awk '{print $2}')

# 如果没有 epoch=*.ckpt（极少情况），回退到 latest.ckpt
if [ -z "${AT_LOAD_DIR}" ]; then
    echo "Warning: No topk ckpt found, falling back to latest.ckpt"
    AT_LOAD_DIR="${AT_CKPT_DIR}/latest.ckpt"
fi

if [ ! -f "${AT_LOAD_DIR}" ]; then
    echo "Error: AT checkpoint not found at ${AT_LOAD_DIR}"
    exit 1
fi
echo "Found AT checkpoint: ${AT_LOAD_DIR}"

# ── Stage 2: 训练 Latent Diffusion Policy ────────────────────────
echo ""
echo "Stage 2: Training Latent Diffusion Policy (LDP)..."
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    task=dobot_rdp_image_tactile_emb_ldp_24fps \
    task.dataset_path=${DATASET_PATH} \
    task.name=dobot_rdp_image_tactile_emb_ldp_24fps_${TIMESTAMP} \
    at=at_dobot_rdp \
    at_load_dir=${AT_LOAD_DIR} \
    logging.mode=${LOGGING_MODE}

echo ""
echo "Training complete!"
