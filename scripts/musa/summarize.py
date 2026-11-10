"""Validate coverage, normalize result IDs in copies, call upstream aggregation."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("1_robot_subject_stability", "2_physical_plausibility", "3_task_adherence_consistency")
TASKS = ("common_manipulation", "long-horizon_planning", "multi-entity_collaboration",
         "spatial_relationship", "visual_reasoning")
ROBOTS = ("dual_arm", "humanoid", "single_arm", "quad")


def canonical(name):
    match = re.fullmatch(r"(?:[\w-]+_)?(\d{4})(?:\.[\w]+)?", str(name))
    if not match:
        raise ValueError("Invalid result ID: " + str(name))
    return match[1]


def valid_number(value, low, high):
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError("Invalid score: " + str(value))


def upstream(relative):
    spec = importlib.util.spec_from_file_location("rbench_upstream_summary", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_csv(source, target, expected, maximum):
    with source.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    ids = []
    for row in rows:
        row["name"] = canonical(row["name"])
        ids.append(row["name"])
        valid_number(row["score"], 1, maximum)
    if len(set(ids)) != len(ids) or set(ids) != expected:
        raise ValueError("Missing, duplicate or unexpected result IDs: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results, model, split, categories, output):
    coverage = {}
    for category in categories:
        marker = results / "input_views" / model / category / "vqa/inputs.json"
        inputs = json.loads(marker.read_text())
        expected = {f"{row['index']:04d}" for row in inputs["videos"]}
        coverage[category] = len(expected)
        if split == "tasks":
            rel = Path(category) / "qwen_local/results.csv"
            copy_csv(results / "5_tasks" / model / rel, output / rel, expected, 5)
            continue
        source = results / "4_embodiments" / model / category
        target = output / category
        for metric in METRICS:
            rel = Path("VQA/qwen_local") / metric / "results.csv"
            copy_csv(source / rel, target / rel, expected, 15 if metric == METRICS[0] else 5)
        motion = json.loads((source / "motion/results.json").read_text())
        ids = []
        for row in motion:
            row["index"] = canonical(row["index"])
            ids.append(row["index"])
            valid_number(row["perceptible_amplitude_robotic_manipulator"], 0, math.inf)
            valid_number(row["motion_smoothness_score"], 0, 1)
        if len(set(ids)) != len(ids) or set(ids) != expected:
            raise ValueError("Motion and VQA must cover exactly the same input IDs")
        (target / "motion").mkdir(exist_ok=True)
        (target / "motion/results.json").write_text(json.dumps(motion, indent=2) + "\n")
    # Run formulas only after all requested categories have passed validation.
    if split == "tasks":
        upstream("eval/5_tasks/summary_scores.py").process_model(str(output), qwen_eval_name="qwen_local")
    else:
        aggregate = upstream("eval/4_embodiments/8_summarize_robot_results.py")
        for category in categories:
            target = output / category
            aggregate.process_single_model(str(target / "VQA/qwen_local"), "qwen", str(target))
        upstream("eval/4_embodiments/summarize_i2v_results.py").merge_robot_scores_single(
            str(output), categories, str(output / "score_summary_qwen.csv"), "qwen")
    (output / "coverage.json").write_text(json.dumps({"model": model, "split": split,
        "sample_counts": coverage, "judge": "qwen_local", "scope": "selected inputs; not automatically a full RBench score"}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("tasks", "embodiments"))
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; raw results are retained")
    args = parser.parse_args()
    allowed = TASKS if args.split == "tasks" else ROBOTS
    categories = args.categories or list(allowed)
    if not categories or len(set(categories)) != len(categories) or any(c not in allowed for c in categories):
        parser.error("choose distinct categories from " + ", ".join(allowed))
    if Path(args.model_name).name != args.model_name or args.model_name in (".", ".."):
        parser.error("model-name must be one directory name")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summarize(args.results_root.resolve(), args.model_name, args.split, categories, args.output_dir.resolve())


if __name__ == "__main__":
    main()
