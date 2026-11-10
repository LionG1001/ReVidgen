"""Create owned symlink views for upstream's numeric VQA / named motion inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def prepare(video_dir, prompt_file, view_dir, motion=False):
    prompts = json.loads(prompt_file.read_text())
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Use the full, ordered upstream category prompt list")
    by_id = {}
    for index, row in enumerate(prompts, 1):
        name = str(row["name"])
        match = re.fullmatch(r"(?:[\w-]+_)?(\d{4})", name)
        if not match or int(match[1]) != index:
            raise ValueError("Prompt indices must retain their original 1-based order: " + name)
        by_id[index] = name
    videos = {}
    for path in sorted(video_dir.glob("*.mp4")):
        match = re.fullmatch(r"(?:[\w-]+_)?(\d{4})", path.stem)
        if not match or int(match[1]) not in by_id:
            raise ValueError("Video has no matching prompt: " + str(path))
        index = int(match[1])
        resolved = path.resolve(strict=True)
        if index in videos and videos[index] != resolved:
            raise ValueError("Different videos share the same prompt index: " + str(index))
        videos[index] = resolved
    if not videos:
        raise ValueError("No .mp4 inputs in " + str(video_dir))
    entries = []
    for index, path in sorted(videos.items()):
        stat = path.stat()
        if stat.st_size == 0:
            raise ValueError("Empty video: " + str(path))
        entries.append({"index": index, "source": str(path), "bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "name": (by_id[index] if motion else f"{index:04d}") + ".mp4"})
    record = {"prompt_sha256": hashlib.sha256(prompt_file.read_bytes()).hexdigest(), "videos": entries}
    marker = view_dir / "inputs.json"
    if view_dir.exists():
        if not marker.exists() or json.loads(marker.read_text()) != record:
            raise ValueError("Use a fresh output/model tag for different inputs: " + str(view_dir))
    else:
        view_dir.mkdir(parents=True)
        (view_dir / "videos").mkdir()
        for row in entries:
            (view_dir / "videos" / row["name"]).symlink_to(row["source"])
        marker.write_text(json.dumps(record, indent=2) + "\n")
    if {p.name for p in (view_dir / "videos").iterdir()} != {r["name"] for r in entries}:
        raise ValueError("Input view contains unexpected entries: " + str(view_dir))
    for row in entries:
        link = view_dir / "videos" / row["name"]
        if not link.is_symlink() or link.resolve() != Path(row["source"]):
            raise ValueError("Input view changed: " + str(link))
    print(f"Prepared {len(entries)}/{len(prompts)} inputs in {view_dir / 'videos'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, required=True)
    parser.add_argument("--motion", action="store_true")
    args = parser.parse_args()
    prepare(args.video_dir.resolve(), args.prompt_file.resolve(), args.view_dir.resolve(), args.motion)


if __name__ == "__main__":
    main()
