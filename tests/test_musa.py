"""CPU-only checks for bootstrap order, dependency isolation and input/score coverage."""
import ast
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "b03df27f0376faa148dcd8cd620a1989a32ca979"


def load(name):
    spec = importlib.util.spec_from_file_location("rbench_musa_" + name, ROOT / f"scripts/musa/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeTests(unittest.TestCase):
    def test_adapter_import_precedes_torch_and_sees_required_environment(self):
        module = load("run")
        seen = []
        torch = SimpleNamespace(musa=SimpleNamespace(is_available=lambda: True),
                                cuda=SimpleNamespace(is_available=lambda: True))

        def importing(name):
            seen.append(name)
            self.assertEqual(os.environ["TORCHADA_FORCE_CUDA_AVAILABLE"], "1")
            self.assertEqual(os.environ["TORCHADA_PLATFORM"], "musa")
            return SimpleNamespace(is_musa_platform=lambda: True) if name == "torchada" else torch

        with patch.dict(os.environ, {}, clear=True), patch.object(module.importlib, "import_module", importing):
            self.assertIs(module.initialize_musa(), torch)
        self.assertEqual(seen, ["torchada", "torch"])

    def test_conflicting_platform_fails_before_import(self):
        module = load("run")
        with patch.dict(os.environ, {"TORCHADA_PLATFORM": "cuda"}, clear=True), \
                patch.object(module.importlib, "import_module") as importing:
            with self.assertRaisesRegex(RuntimeError, "TORCHADA_PLATFORM"):
                module.initialize_musa()
            importing.assert_not_called()

    def test_missing_adapter_has_install_instructions(self):
        module = load("run")
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(module.importlib, "import_module", side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, "install.py"):
                module.initialize_musa()

    def test_no_cpu_fallback(self):
        module = load("run")
        torch = SimpleNamespace(musa=SimpleNamespace(is_available=lambda: False))
        with patch.dict(os.environ, {}, clear=True), patch.object(module.importlib, "import_module", side_effect=[
                SimpleNamespace(is_musa_platform=lambda: True), torch]):
            with self.assertRaisesRegex(RuntimeError, "working torch_musa"):
                module.initialize_musa()

    def test_installer_preview_never_installs_or_builds(self):
        module = load("install")
        with patch.object(module, "gpu_stack", return_value={"torch": "2.11.0.post1+musa5.2"}), \
                patch.object(module.subprocess, "run") as run, patch.dict(os.environ, {}, clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            module.main(["ops", "--dry-run"])
        run.assert_not_called()
        self.assertIn("--no-build-isolation", output.getvalue())
        self.assertIn("disposable-pkgs-copy", output.getvalue())


class InputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.videos = self.root / "source"
        self.videos.mkdir()
        self.prompts = self.root / "prompts.json"
        self.prompts.write_text(json.dumps([{"name": f"single_arm_{i:04d}"} for i in range(1, 5)]))
        # Placeholder bytes test path handling only; no video decode is claimed.
        (self.videos / "single_arm_0002.mp4").write_bytes(b"video-2")
        (self.videos / "0004.mp4").write_bytes(b"video-4")

    def test_noncontiguous_subset_has_both_upstream_name_views(self):
        prepare = load("prepare_inputs").prepare
        for motion, names in [(False, ["0002.mp4", "0004.mp4"]),
                              (True, ["single_arm_0002.mp4", "single_arm_0004.mp4"])]:
            target = self.root / str(motion)
            prepare(self.videos, self.prompts, target, motion)
            self.assertEqual(sorted(p.name for p in (target / "videos").iterdir()), names)
            self.assertTrue(all(p.is_symlink() for p in (target / "videos").iterdir()))
            prepare(self.videos, self.prompts, target, motion)  # Owned, unchanged view can be reused.

    def test_changed_inputs_cannot_reuse_cached_grids(self):
        prepare = load("prepare_inputs").prepare
        target = self.root / "view"
        prepare(self.videos, self.prompts, target)
        (self.videos / "0004.mp4").write_bytes(b"different video")
        with self.assertRaisesRegex(ValueError, "fresh output"):
            prepare(self.videos, self.prompts, target)

    def test_two_videos_cannot_share_one_prompt_id(self):
        (self.videos / "0002.mp4").write_bytes(b"other video")
        with self.assertRaisesRegex(ValueError, "Different videos"):
            load("prepare_inputs").prepare(self.videos, self.prompts, self.root / "view")

    def test_prompt_subset_must_not_reindex_original_positions(self):
        self.prompts.write_text('[{"name":"single_arm_0002"}]')
        with self.assertRaisesRegex(ValueError, "original 1-based"):
            load("prepare_inputs").prepare(self.videos, self.prompts, self.root / "view")

    def test_empty_video_fails_before_scoring(self):
        (self.videos / "0004.mp4").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Empty video"):
            load("prepare_inputs").prepare(self.videos, self.prompts, self.root / "view")

    def test_extra_file_in_owned_view_is_rejected(self):
        prepare = load("prepare_inputs").prepare
        target = self.root / "view"
        prepare(self.videos, self.prompts, target)
        (target / "videos/0001.mp4").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "unexpected entries"):
            prepare(self.videos, self.prompts, target)


class LauncherTests(unittest.TestCase):
    def commands(self, script, **values):
        env = dict(os.environ, RBENCH_DRY_RUN="1", RBENCH_MODEL_NAME="H20 step500 EMA",
                   QWEN_MODEL_PATH="/models/Qwen judge", **values)
        result = subprocess.run(["bash", str(ROOT / "scripts" / script)], cwd="/tmp", env=env,
                                text=True, capture_output=True, check=True)
        return [shlex.split(line) for line in result.stdout.splitlines()]

    def test_all_five_tasks_use_local_judge_and_keep_spaced_paths(self):
        commands = self.commands("rbench_eval_5tasks_musa.sh")
        scorers = [c for c in commands if c[1].endswith("/musa/run.py")]
        self.assertEqual(len(scorers), 5)
        for command in scorers:
            self.assertEqual(command[command.index("--model") + 1], "qwen_local")
            self.assertEqual(command[command.index("--qwen_model_path") + 1], "/models/Qwen judge")
            self.assertNotIn("--api_key", command)

    def test_embodiment_profiles_do_not_mix_vlm_and_ops(self):
        for stage, scorer_count in [("vqa", 12), ("motion", 8)]:
            commands = self.commands("rbench_eval_4embodiments_musa.sh", RBENCH_STAGE=stage)
            scorers = [c for c in commands if c[1].endswith("/musa/run.py")]
            self.assertEqual(len(scorers), scorer_count)
            self.assertTrue(all(("--qwen_model_path" in c) == (stage == "vqa") for c in scorers))


class SummaryTests(unittest.TestCase):
    def test_id_normalization_and_original_penalty_formula(self):
        module = load("summarize")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            marker = root / "input_views/model/single_arm/vqa/inputs.json"
            marker.parent.mkdir(parents=True)
            marker.write_text('{"videos":[{"index":1},{"index":2}]}')
            source = root / "4_embodiments/model/single_arm"
            for metric, scores in zip(module.METRICS, [(15, 8), (5, 3), (5, 3)]):
                p = source / "VQA/qwen_local" / metric / "results.csv"
                p.parent.mkdir(parents=True)
                p.write_text(f'name,score,option\n0001.jpeg,{scores[0]},"A1,A2"\nsingle_arm_0002,{scores[1]},"A1,A2"\n')
            motion = source / "motion/results.json"
            motion.parent.mkdir()
            motion.write_text(json.dumps([{"index": f"single_arm_{i:04d}",
                "perceptible_amplitude_robotic_manipulator": .3, "motion_smoothness_score": score}
                for i, score in [(1, 1), (2, .5)]]))
            raw = motion.read_bytes()
            output = root / "summary"
            module.summarize(root, "model", "embodiments", ["single_arm"], output)
            self.assertEqual(motion.read_bytes(), raw)
            with (output / "single_arm/score_summary_penalized_qwen.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[-1]["name"], "MEAN")
            self.assertEqual(float(rows[-1]["Task_Completion"]), .75)
            self.assertEqual(float(rows[-1]["Visual_Quality"]), .75)
            motion.write_text('[{"index":"single_arm_0001","perceptible_amplitude_robotic_manipulator":0.3,"motion_smoothness_score":1}]')
            with self.assertRaisesRegex(ValueError, "exactly the same"):
                module.summarize(root, "model", "embodiments", ["single_arm"], root / "bad")

    def test_failed_judge_response_is_not_dropped_from_denominator(self):
        module = load("summarize")
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "raw.csv"
            p.write_text("name,score\n0001,-1\n")
            with self.assertRaisesRegex(ValueError, "Invalid score"):
                module.copy_csv(p, Path(name) / "copy.csv", {"0001"}, 5)


class UpstreamParityTests(unittest.TestCase):
    def test_scorer_functions_only_move_unused_video_writer_import(self):
        class StripImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.module == "torchvision.io" else node

        files = list((ROOT / "eval/5_tasks").glob("*.py")) + [ROOT / "eval/4_embodiments" / f
            for f in ("1_robot_subject_stability.py", "2_physical_plausibility.py", "3_task_adherence_consistency.py",
                      "5_motion_amplitude.py", "6_motion_smoothness.py")]
        for path in files:
            relative = path.relative_to(ROOT)
            original = subprocess.check_output(["git", "show", f"{BASELINE}:{relative}"], cwd=ROOT, text=True)
            definitions = []
            for text in (original, path.read_text()):
                nodes = ast.parse(text).body
                definitions.append({n.name: ast.dump(StripImports().visit(n), include_attributes=False)
                                    for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))})
            self.assertEqual(definitions[0], definitions[1], str(relative))


if __name__ == "__main__":
    unittest.main()
