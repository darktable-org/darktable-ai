#!/bin/bash
# Shared functions for model setup, conversion, and cleanup scripts.
#
# Usage: source this file after setting SCRIPT_DIR, VENV_DIR, and
# sourcing the model's model.conf.
#
# Expected variables from model.conf:
#   MODEL_ID              - e.g. "mask-object-samhq-light"
#   REPO_URL              - git clone URL
#   REPO_DIR              - local directory name for the cloned repo
#   REPO_INSTALL_CMD      - command to install the repo (run from repo dir)
#   REPO_HAS_REQUIREMENTS - "true" to pip install -r requirements.txt in repo
#   CHECKPOINT_URLS[]     - array of download URLs (supports direct URLs,
#                           Google Drive URLs, or gdrive://FILE_ID)
#   CHECKPOINT_PATHS[]    - array of paths relative to ROOT_DIR
#   MODEL_TYPE            - "single" (default) or "split" (encoder + decoder)
#
# Optional variables for pre-converted ONNX models (no conversion needed):
#   ONNX_URLS[]           - array of pre-converted ONNX download URLs
#   ONNX_PATHS[]          - array of output paths relative to ROOT_DIR
#
# Optional functions from model.conf:
#   run_convert()         - conversion command (omit for pre-converted models)
#   demo_args()           - per-image demo arguments (e.g. point prompts)

ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

setup_venv() {
    if [ -d "$VENV_DIR" ]; then
        echo "Virtual environment already exists at $VENV_DIR"
    else
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing project dependencies..."
    pip install -r "$SCRIPT_DIR/requirements.txt"

    # Install optional dependencies (best-effort, one at a time).
    if [ -f "$SCRIPT_DIR/optional-requirements.txt" ]; then
        echo "Installing optional dependencies (failures are non-fatal)..."
        while IFS= read -r pkg || [ -n "$pkg" ]; do
            pkg="${pkg%%#*}"       # strip comments
            pkg="${pkg// /}"       # strip whitespace
            [ -z "$pkg" ] && continue
            pip install "$pkg" || echo "  Warning: optional package '$pkg' failed to install, skipping."
        done < "$SCRIPT_DIR/optional-requirements.txt"
    fi
}

clone_and_install_repo() {
    if ! [ -d "$SCRIPT_DIR/$REPO_DIR" ]; then
        echo "Cloning $REPO_DIR..."
        git clone "$REPO_URL" "$SCRIPT_DIR/$REPO_DIR"
    else
        echo "$REPO_DIR already cloned at $SCRIPT_DIR/$REPO_DIR"
    fi

    cd "$SCRIPT_DIR/$REPO_DIR"
    if [ "${REPO_HAS_REQUIREMENTS:-false}" = true ]; then
        pip install -r requirements.txt
    fi
    eval "$REPO_INSTALL_CMD"
    cd "$SCRIPT_DIR"
}

download_gdrive() {
    local file_id="$1"
    local output="$2"
    curl -L -o "$output" \
        "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"
}

# Extract Google Drive file ID from various URL formats:
#   https://drive.google.com/file/d/FILE_ID/view
#   https://drive.google.com/uc?id=FILE_ID
#   https://drive.google.com/open?id=FILE_ID
#   gdrive://FILE_ID
gdrive_file_id() {
    local url="$1"
    if [[ "$url" =~ ^gdrive://(.+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$url" =~ /file/d/([^/]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$url" =~ [\?\&]id=([^\&]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    fi
}

download_checkpoints() {
    for i in "${!CHECKPOINT_URLS[@]}"; do
        local url="${CHECKPOINT_URLS[$i]}"
        local path="$ROOT_DIR/${CHECKPOINT_PATHS[$i]}"
        if ! [ -f "$path" ]; then
            echo "Downloading checkpoint: $(basename "$path")..."
            mkdir -p "$(dirname "$path")"
            local gdrive_id
            gdrive_id=$(gdrive_file_id "$url")
            if [ -n "$gdrive_id" ]; then
                download_gdrive "$gdrive_id" "$path"
            else
                curl -L "$url" -o "$path"
            fi
        else
            echo "Checkpoint already exists at $path"
        fi
    done
}

download_onnx_models() {
    for i in "${!ONNX_URLS[@]}"; do
        local url="${ONNX_URLS[$i]}"
        local path="$ROOT_DIR/${ONNX_PATHS[$i]}"
        if ! [ -f "$path" ]; then
            echo "Downloading ONNX model: $(basename "$path")..."
            mkdir -p "$(dirname "$path")"
            curl -L "$url" -o "$path"
        else
            echo "ONNX model already exists at $path"
        fi
    done
}

activate_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Error: Virtual environment not found. Run 'run.sh $MODEL_ID setup' first."
        exit 1
    fi
    source "$VENV_DIR/bin/activate"
}

# Auto-detect setup steps from model.conf variables.
setup_env() {
    setup_venv
    if [ -n "$REPO_URL" ]; then
        clone_and_install_repo
    fi
    if [ ${#CHECKPOINT_URLS[@]} -gt 0 ]; then
        download_checkpoints
    fi
    if [ ${#ONNX_URLS[@]} -gt 0 ]; then
        download_onnx_models
    fi
    echo "Success! Environment ready at $VENV_DIR"
}

# Generate config.json in the output directory from model.conf variables.
generate_config() {
    local output_dir="$ROOT_DIR/output/$MODEL_ID"
    local config_file="$output_dir/config.json"
    mkdir -p "$output_dir"
    cat > "$config_file" <<EOF
{
    "id": "$MODEL_ID",
    "name": "$MODEL_NAME",
    "description": "$MODEL_DESCRIPTION",
    "task": "$MODEL_TASK",
    "backend": "onnx",
    "version": "1.0",
    "tiling": ${MODEL_TILING:-false}
}
EOF
    echo "Generated: $config_file"
}

# Run model conversion (calls run_convert from model.conf if defined).
run_conversion() {
    if declare -f run_convert > /dev/null; then
        activate_venv
        run_convert
    else
        echo "Pre-converted ONNX model — no conversion needed."
    fi
    generate_config
}

# Package output directory as a .dtmodel file (zip archive).
package_model() {
    local output_dir="$ROOT_DIR/output/$MODEL_ID"
    local package="$ROOT_DIR/output/$MODEL_ID.dtmodel"
    if [ ! -d "$output_dir" ]; then
        echo "Error: Output directory not found at $output_dir"
        exit 1
    fi
    echo "Packaging $MODEL_ID.dtmodel..."
    (cd "$ROOT_DIR/output" && zip -r "$MODEL_ID.dtmodel" "$MODEL_ID/")
    echo "Created: $package"
}

# Validate ONNX output: check files load and print I/O shapes.
run_validation() {
    activate_venv
    local model_dir="$ROOT_DIR/output/$MODEL_ID"
    python3 "$MODELS_DIR/validate.py" \
        --model-dir "$model_dir" \
        --model-type "${MODEL_TYPE:-single}"
}

# Run demo on sample images with correct model args.
run_demo_pipeline() {
    local model_dir="$ROOT_DIR/output/$MODEL_ID"
    if [ "${MODEL_TYPE:-single}" = "split" ]; then
        run_demo --encoder "$model_dir/encoder.onnx" --decoder "$model_dir/decoder.onnx"
    else
        run_demo --model "$model_dir/model.onnx"
    fi
}

run_demo() {
    # Run demo.py on all sample images.
    # Usage: run_demo [extra args for demo.py...]
    # Automatically provides --image and --output for each sample image.
    # Model-specific args (e.g. --model, --encoder) are passed through.
    #
    # If the calling script defines a demo_args() function, it is called
    # with the image basename (e.g. "example_01") and its output is appended
    # as extra arguments. Use this for per-image inputs like point prompts.
    activate_venv

    local images_dir="$ROOT_DIR/images"
    local output_dir="$SCRIPT_DIR/output"
    mkdir -p "$output_dir"

    for img in "$images_dir"/example_*.jpg "$images_dir"/example_*.png; do
        [ -f "$img" ] || continue
        local name
        name="$(basename "${img%.*}")"
        local output="$output_dir/${name}.png"

        local extra=""
        if declare -f demo_args > /dev/null; then
            extra="$(demo_args "$name")"
        fi

        echo "  $name"
        # shellcheck disable=SC2086
        python3 "$SCRIPT_DIR/demo.py" \
            "$@" \
            $extra \
            --image "$img" \
            --output "$output"
    done
}

clean_env() {
    echo "Cleaning up environment for $MODEL_ID..."

    if [ -d "$VENV_DIR" ]; then
        echo "Removing virtual environment at $VENV_DIR..."
        rm -rf "$VENV_DIR"
    fi

    if [ -n "$REPO_DIR" ] && [ -d "$SCRIPT_DIR/$REPO_DIR" ]; then
        echo "Removing $REPO_DIR repository at $SCRIPT_DIR/$REPO_DIR..."
        rm -rf "$SCRIPT_DIR/$REPO_DIR"
    fi

    echo "Cleanup complete."
}
