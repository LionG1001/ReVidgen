#!/usr/bin/env bash
# Sourced by the two local-Qwen/MUSA launchers.
set -euo pipefail
RBENCH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
RBENCH_DRY_RUN=${RBENCH_DRY_RUN:-0}
: "${RBENCH_MODEL_NAME:?Set a unique checkpoint/EMA/seed tag in RBENCH_MODEL_NAME}"
[[ "$RBENCH_MODEL_NAME" != */* && "$RBENCH_MODEL_NAME" != . && "$RBENCH_MODEL_NAME" != .. ]] || {
    echo "RBENCH_MODEL_NAME must be one directory name" >&2; exit 2;
}
cd "$RBENCH_ROOT"
RBENCH_VIDEO_ROOT=${RBENCH_VIDEO_ROOT:-$RBENCH_ROOT/data/$RBENCH_MODEL_NAME}
RBENCH_PROMPT_ROOT=${RBENCH_PROMPT_ROOT:-$RBENCH_ROOT/data/prompts}
RBENCH_RESULTS_ROOT=${RBENCH_RESULTS_ROOT:-$RBENCH_ROOT/results}
export MPLBACKEND=Agg

run() {
    if [[ "$RBENCH_DRY_RUN" == 1 ]]; then
        printf '%q ' "$@"; printf '\n'
    else
        "$@"
    fi
}

scorer() { run "$PYTHON_BIN" "$RBENCH_ROOT/scripts/musa/run.py" "$@"; }

prepare_inputs() {
    local category=$1 mode=$2
    INPUT_VIEW="$RBENCH_RESULTS_ROOT/input_views/$RBENCH_MODEL_NAME/$category/$mode"
    local args=(--video-dir "$RBENCH_VIDEO_ROOT/$category/videos"
                --prompt-file "$RBENCH_PROMPT_ROOT/${category}_prompts.json" --view-dir "$INPUT_VIEW")
    if [[ "$mode" == motion ]]; then args+=(--motion); fi
    run "$PYTHON_BIN" "$RBENCH_ROOT/scripts/musa/prepare_inputs.py" "${args[@]}"
}
