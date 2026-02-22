#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$ROOT_DIR/models"

# Verify Python version is within supported range (3.11–3.12).
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 11 ] || [ "$PYTHON_MINOR" -gt 12 ]; then
    echo "Error: Python 3.11–3.12 is required (found $PYTHON_VERSION)"
    exit 1
fi

if [ -z "$1" ]; then
    echo "Usage: $0 <model_id> [setup|convert|validate|package|demo|clean]"
    echo ""
    echo "Available models:"
    for dir in "$MODELS_DIR"/*/; do
        if [ -f "$dir/model.conf" ]; then
            echo "  $(basename "$dir")"
        fi
    done
    exit 1
fi

MODEL_ID="$1"
ACTION="${2:-all}"

SCRIPT_DIR="$(cd "$MODELS_DIR/$MODEL_ID" 2>/dev/null && pwd)" || {
    echo "Error: Model '$MODEL_ID' not found in models/"
    exit 1
}

if [ ! -f "$SCRIPT_DIR/model.conf" ]; then
    echo "Error: No model.conf found in $SCRIPT_DIR"
    exit 1
fi

VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/model.conf"
source "$MODELS_DIR/common.sh"

case "$ACTION" in
    setup)
        setup_env
        ;;
    convert)
        run_conversion
        ;;
    validate)
        run_validation
        ;;
    package)
        package_model
        ;;
    demo)
        run_demo_pipeline
        ;;
    clean)
        clean_env
        ;;
    all)
        echo "=== Setup ==="
        setup_env

        echo ""
        echo "=== Convert ==="
        run_conversion

        echo ""
        echo "=== Validate ==="
        run_validation

        echo ""
        echo "=== Package ==="
        package_model

        echo ""
        echo "=== Demo ==="
        run_demo_pipeline

        echo ""
        echo "=== Clean ==="
        clean_env
        ;;
    *)
        echo "Error: Unknown action '$ACTION'. Use: setup, convert, validate, package, demo, clean"
        exit 1
        ;;
esac
