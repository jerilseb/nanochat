#!/bin/bash
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
WANDB_RUN="5090_speedrun" # Set to "dummy" if you don't use weights & biases

# 1. Setup Environment
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# 2. Download Data & Train Tokenizer
python -m nanochat.dataset -n 8
# Kick off downloading the rest of the 170 shards in the background
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# 3. PRETRAINING (The heavy lifting)
# We drop device-batch-size to 8 to fit 32GB VRAM, and set window-pattern to L for SDPA efficiency.
python -m scripts.base_train \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=8 \
    --window-pattern=L \
    --run=$WANDB_RUN

# Evaluate the base model
python -m scripts.base_eval --device-batch-size=8

# 4. SUPERVISED FINE-TUNING (SFT)
# Teach the model how to chat and use tools. Download the identity file first.
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft \
    --device-batch-size=8 \
    --run=$WANDB_RUN

python -m scripts.chat_eval -i sft

# 5. Generate Report
python -m nanochat.report generate