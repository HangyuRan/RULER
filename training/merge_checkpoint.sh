#!/usr/bin/env bash
# Convert a VERL FSDP actor checkpoint to Hugging Face format.
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "Usage: bash $0 <actor_checkpoint_dir> <output_model_dir>" >&2
    exit 2
fi

SOURCE_DIR="$1"
TARGET_DIR="$2"

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: source directory does not exist: $SOURCE_DIR" >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$SOURCE_DIR" \
    --target_dir "$TARGET_DIR"

echo "Merge completed: $TARGET_DIR"
ls -lh "$TARGET_DIR"
