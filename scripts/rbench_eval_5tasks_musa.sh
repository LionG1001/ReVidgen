#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/musa/common.sh"
: "${QWEN_MODEL_PATH:?Set a local Qwen2.5-VL-72B-Instruct judge path}"
read -r -a tasks <<< "${RBENCH_TASKS:-common_manipulation long-horizon_planning multi-entity_collaboration spatial_relationship visual_reasoning}"
for task in "${tasks[@]}"; do
    case "$task" in
        common_manipulation|long-horizon_planning|multi-entity_collaboration|visual_reasoning) ;;
        spatial_relationship) ;;
        *) echo "Unknown task: $task" >&2; exit 2 ;;
    esac
    prepare_inputs "$task" vqa
    scorer "eval/5_tasks/$task.py" --model qwen_local \
        --video_path "$INPUT_VIEW/videos" \
        --read_prompt_file "$RBENCH_PROMPT_ROOT/${task}_prompts.json" \
        --output_path "$RBENCH_RESULTS_ROOT/5_tasks/$RBENCH_MODEL_NAME/$task/qwen_local" \
        --qwen_model_path "$QWEN_MODEL_PATH" --max_new_tokens 1024
done
