#!/bin/bash
# Evaluate a model on a standard benchmark.
#
# Usage: ./eval.sh <task> <model_id> [extra args...]
# Example: ./eval.sh mask mask-object-segnext-b2hq --limit 5
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$ROOT_DIR/evaluation"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <task> <model_id> [extra args...]"
    echo ""
    echo "Available tasks:"
    for dir in "$EVAL_DIR"/*/; do
        [ -f "$dir/evaluate.py" ] && echo "  $(basename "$dir")"
    done
    exit 1
fi

TASK="$1"
MODEL_ID="$2"
shift 2

TASK_DIR="$EVAL_DIR/$TASK"
VENV_DIR="$TASK_DIR/.venv"
MODEL_DIR="$ROOT_DIR/output/$MODEL_ID"

if [ ! -f "$TASK_DIR/evaluate.py" ]; then
    echo "Error: No evaluate.py found for task '$TASK' in $TASK_DIR"
    exit 1
fi

# Resolve encoder/decoder paths
ENCODER="$MODEL_DIR/encoder.onnx"
DECODER="$MODEL_DIR/decoder.onnx"

if [ ! -f "$ENCODER" ] || [ ! -f "$DECODER" ]; then
    echo "Error: ONNX models not found in $MODEL_DIR"
    echo "  Expected: encoder.onnx and decoder.onnx"
    echo "  Run: ./run.sh $MODEL_ID convert"
    exit 1
fi

# Setup venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating evaluation venv..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r "$TASK_DIR/requirements.txt"
else
    source "$VENV_DIR/bin/activate"
fi

# Download DAVIS dataset if needed (for mask task)
DATASET_PATH="$ROOT_DIR/temp/DAVIS"
if [ "$TASK" = "mask" ] && [ ! -d "$DATASET_PATH/JPEGImages" ]; then
    echo "Downloading DAVIS-2017-trainval-480p..."
    mkdir -p "$ROOT_DIR/temp"
    DAVIS_ZIP="$ROOT_DIR/temp/davis-2017-trainval-480p.zip"
    if [ ! -f "$DAVIS_ZIP" ]; then
        curl -L -o "$DAVIS_ZIP" \
            "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
    fi
    echo "Extracting..."
    unzip -q "$DAVIS_ZIP" -d "$ROOT_DIR/temp"
    echo "Done."
fi

# Run evaluation
python3 "$TASK_DIR/evaluate.py" \
    --encoder "$ENCODER" \
    --decoder "$DECODER" \
    --dataset-path "$DATASET_PATH" \
    "$@"
