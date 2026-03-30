#!/bin/bash
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
WANDB_RUN="Single_H100_speedrun" # Set to "dummy" if you don't use weights & biases

# 1. Setup Environment
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# 2. Train Tokenizer (ONLY if it doesn't exist)
TOKENIZER_FILE="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"

if [ ! -f "$TOKENIZER_FILE" ]; then
    echo "Tokenizer not found. Downloading initial data and training tokenizer..."
    python -m nanochat.dataset -n 8
    python -m scripts.tok_train
    python -m scripts.tok_eval
else
    echo "✅ Tokenizer already exists at $TOKENIZER_FILE. Skipping training..."
fi

# Ensure all 170 pretraining shards are downloaded.
echo "Ensuring all 170 pretraining data shards are present..."
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

echo "Waiting for any pending dataset downloads to complete..."
wait $DATASET_DOWNLOAD_PID

# 3. PRETRAINING (The heavy lifting)
echo "🚀 Starting base model pretraining..."
# --device-batch-size=16 perfectly fits in 80GB VRAM.
# --fp8 enables 8-bit precision for a massive speedup on H100.
python -m scripts.base_train \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=16 \
    --fp8 \
    --run=$WANDB_RUN

# Evaluate the base model
echo "📊 Evaluating base model..."
python -m scripts.base_eval --device-batch-size=16

# 4. SUPERVISED FINE-TUNING (SFT)
echo "🗣️ Starting SFT (Teaching the model how to chat)..."

IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ ! -f "$IDENTITY_FILE" ]; then
    echo "Downloading identity dataset..."
    curl -L -o "$IDENTITY_FILE" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
else
    echo "✅ Identity dataset already exists. Skipping download..."
fi

# Note: SFT does not use FP8 by default in this codebase to ensure the highest quality chat alignment
python -m scripts.chat_sft \
    --device-batch-size=16 \
    --run=$WANDB_RUN

echo "📊 Evaluating chat model..."
python -m scripts.chat_eval -i sft

# 5. Generate Report
echo "📝 Generating final markdown report..."
python -m nanochat.report generate