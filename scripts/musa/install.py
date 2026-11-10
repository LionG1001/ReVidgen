"""Install a RBench MUSA dependency profile into the container's default Python."""
import argparse
import datetime
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
STACK = ("torch", "torch_musa", "torchvision", "torchaudio", "triton")
PACKAGES = ("Grounded-Segment-Anything/segment_anything", "Grounded-Segment-Anything/GroundingDINO",
            "sam2", "Q-Align", "co-tracker")


def gpu_stack():
    versions = {}
    for name in STACK:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            if name in ("torch", "torch_musa", "torchvision"):
                raise RuntimeError(f"Preinstall the matching vendor {name} package") from None
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("vlm", "ops"), help="Use different containers for the two profiles")
    parser.add_argument("--dry-run", action="store_true", help="Print installation commands; do not install or build")
    args = parser.parse_args(argv)
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        parser.error("these profiles target Python 3.10 through 3.12")
    if sys.prefix != sys.base_prefix or os.environ.get("VIRTUAL_ENV"):
        parser.error("use the container's default Python, outside venv")
    stack = gpu_stack()  # Read package metadata only; no GPU initialization.
    env = dict(os.environ, RBENCH_USE_TORCHADA="1", TORCHADA_PLATFORM="musa",
               TORCHADA_FORCE_CUDA_AVAILABLE="1", MPLBACKEND="Agg",
               SAM2_BUILD_ALLOW_ERRORS="0", AM_I_DOCKER="1", BUILD_WITH_CUDA="1")
    env.setdefault("MAX_JOBS", "8")
    requirements = ROOT / "environment/musa" / f"requirements-{args.profile}.txt"
    pip = [sys.executable, "-m", "pip", "install"]
    if args.dry_run:
        print(json.dumps({"preserved_gpu_stack": stack, "requirements": str(requirements)}, indent=2))
        print(shlex.join([*pip, "-c", "<generated-installed-gpu-stack.txt>", "-r", str(requirements)]))
        if args.profile == "ops":
            for package in PACKAGES:
                print(shlex.join([*pip, "--no-deps", "--no-build-isolation", "<disposable-pkgs-copy>/" + package]))
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    build = Path(tempfile.mkdtemp(prefix=stamp + "-", dir=_build_parent()))
    constraints = build / "installed-gpu-stack.txt"
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in stack.items()))
    # PIP_CONSTRAINT also protects any child pip invocation from selecting CUDA Torch.
    env["PIP_CONSTRAINT"] = str(constraints)
    subprocess.run([*pip, "-c", str(constraints), "-r", str(requirements)], env=env, check=True)
    if args.profile == "ops":
        # torchada rewrites portable C++ sources in place during compilation.
        # Only mutate this disposable copy; the reviewed Git checkout stays intact.
        copied = build / "pkgs"
        shutil.copytree(ROOT / "pkgs", copied,
                        ignore=shutil.ignore_patterns(".git", "__pycache__", "build", "*.so", "*.egg-info",
                                                     "checkpoints", "*.pt", "*.pth", "*.safetensors"))
        for package in PACKAGES:
            subprocess.run([*pip, "--no-deps", "--no-build-isolation", str(copied / package)], env=env, check=True)
        # Do not report a successful installation after an optional/CPU-only build.
        subprocess.run([sys.executable, "-c", "import torchada; import groundingdino._C; "
                        "import sam2._C; from cotracker.predictor import CoTrackerPredictor; "
                        "from q_align.model.builder import load_pretrained_model"], env=env, check=True)
    if gpu_stack() != stack:
        raise RuntimeError("GPU stack changed during installation; see " + str(constraints))
    manifest = {"profile": args.profile, "gpu_stack": stack, "python": sys.executable,
                "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "requirements": str(requirements), "build": str(build)}
    (build / "install-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (build / "freeze.txt").write_text("".join(sorted(
        f"{dist.metadata['Name']}=={dist.version}\n" for dist in metadata.distributions()
        if dist.metadata.get("Name"))))
    print("Installed", args.profile, "profile; manifest:", build / "install-manifest.json")


def _build_parent():
    parent = ROOT / ".musa-install"
    parent.mkdir(exist_ok=True)
    return parent


if __name__ == "__main__":
    main()
