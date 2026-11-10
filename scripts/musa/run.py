"""Import torchada before running an upstream RBench scorer and its dependencies."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]


def initialize_musa():
    # torchada reads these once on import. Reject a conflicting launch setup.
    for key, value in {"TORCHADA_PLATFORM": "musa", "TORCHADA_FORCE_CUDA_AVAILABLE": "1"}.items():
        if os.environ.get(key, value) != value:
            raise RuntimeError(f"{key} must be {value!r} for the MUSA entry point")
        os.environ[key] = value
    try:
        adapter = importlib.import_module("torchada")
    except ImportError as exc:
        raise RuntimeError("Install the MUSA profile with scripts/musa/install.py first") from exc
    torch = importlib.import_module("torch")
    if not adapter.is_musa_platform() or not torch.musa.is_available():
        raise RuntimeError("The MUSA scorer requires a working torch_musa device")
    if not torch.cuda.is_available():
        raise RuntimeError("torchada was imported before TORCHADA_FORCE_CUDA_AVAILABLE=1 was set")
    return torch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the backend without loading judge weights")
    parser.add_argument("script", nargs="?", help="Upstream eval/... Python script")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.check and args.script is None:
        parser.error("provide an eval script or --check")
    target = None
    if args.script:
        target = (ROOT / args.script).resolve()
        if not target.is_relative_to(ROOT / "eval") or target.suffix != ".py" or not target.is_file():
            parser.error("script must be an existing Python file under this checkout's eval/")
    torch = initialize_musa()
    if args.check:
        print(json.dumps({"torch": torch.__version__, "torch_musa": importlib.metadata.version("torch_musa"),
                          "torchada": importlib.metadata.version("torchada"),
                          "musa_devices": torch.musa.device_count(),
                          "cuda_api_maps_to": str(torch.device("cuda"))}, indent=2))
        return
    os.chdir(ROOT)  # Upstream motion scripts resolve config/checkpoints relative to the repo.
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *args.script_args]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
