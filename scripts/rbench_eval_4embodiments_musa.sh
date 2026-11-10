#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/musa/common.sh"
RBENCH_STAGE=${RBENCH_STAGE:-vqa}
case "$RBENCH_STAGE" in
    vqa) : "${QWEN_MODEL_PATH:?Set a local Qwen2.5-VL-72B-Instruct judge path}" ;;
    motion) ;;
    *) echo "RBENCH_STAGE must be vqa or motion" >&2; exit 2 ;;
esac
read -r -a robots <<< "${RBENCH_ROBOTS:-dual_arm humanoid single_arm quad}"
for robot in "${robots[@]}"; do
    case "$robot" in dual_arm|humanoid|single_arm|quad) ;; *) echo "Unknown robot: $robot" >&2; exit 2 ;; esac
    output="$RBENCH_RESULTS_ROOT/4_embodiments/$RBENCH_MODEL_NAME/$robot"
    if [[ "$RBENCH_STAGE" == vqa ]]; then
        prepare_inputs "$robot" vqa
        for metric in 1_robot_subject_stability 2_physical_plausibility 3_task_adherence_consistency; do
            scorer "eval/4_embodiments/$metric.py" --model qwen_local \
                --video_path "$INPUT_VIEW/videos" \
                --read_prompt_file "$RBENCH_PROMPT_ROOT/${robot}_prompts.json" \
                --output_path "$output/VQA/qwen_local/$metric" \
                --qwen_model_path "$QWEN_MODEL_PATH" --max_new_tokens 1024
        done
    else
        prepare_inputs "$robot" motion
        meta="$output/motion/results.json"
        run "$PYTHON_BIN" eval/4_embodiments/4_create_meta_info.py \
            -v "$INPUT_VIEW/videos" -i "$RBENCH_PROMPT_ROOT/${robot}_prompts.json" -o "$meta"
        scorer eval/4_embodiments/5_motion_amplitude.py --meta_info_path "$meta" \
            --target_type robotic_manipulator --box_threshold 0.25 --text_threshold 0.20 --grid_size 30 --device cuda
        scorer eval/4_embodiments/6_motion_smoothness.py --meta_info_path "$meta" --device cuda
        run "$PYTHON_BIN" eval/4_embodiments/7_motion_total_score.py --meta_info_path "$meta"
    fi
done
