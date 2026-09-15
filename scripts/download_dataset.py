"""Download the car damage dataset from Kaggle and (optionally) convert it
to Ultralytics YOLO-seg layout.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --slug owner/dataset-name --skip-convert

Auth: relies on kagglehub's standard credential resolution — set
KAGGLE_USERNAME/KAGGLE_KEY (in .env or the shell environment) or place a
kaggle.json under ~/.kaggle/. No custom credential handling is done here.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings


def download_raw(slug: str, dest_dir: Path) -> Path:
    import kagglehub

    dest_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(kagglehub.dataset_download(slug))

    for item in cache_path.iterdir():
        target = dest_dir / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    return dest_dir


def report_contents(raw_dir: Path) -> None:
    print(f"\nContents of {raw_dir}:")
    entries = sorted(raw_dir.rglob("*"))
    for entry in entries[:50]:
        kind = "dir" if entry.is_dir() else "file"
        print(f"  [{kind}] {entry.relative_to(raw_dir)}")
    if len(entries) > 50:
        print(f"  ... and {len(entries) - 50} more entries")
    print(
        "\nInspect the structure above to confirm whether it is already "
        "YOLO-shaped (images/ + labels/*.txt) or COCO-shaped (a *.json with "
        "images/annotations/categories keys). Adjust training/convert_to_yolo.py "
        "if the detected format doesn't match what convert_to_yolo.py expects.\n"
    )


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=settings.dataset_slug, help="Kaggle dataset slug (owner/name)")
    parser.add_argument("--raw-dir", default=str(settings.dataset_raw_dir), help="Where to store raw downloaded data")
    parser.add_argument(
        "--processed-dir",
        default=str(settings.dataset_processed_dir),
        help="Where the converted YOLO-seg layout will be written",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Only download raw data, skip the YOLO-seg conversion step",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    print(f"Downloading Kaggle dataset '{args.slug}' into {raw_dir} ...")
    download_raw(args.slug, raw_dir)
    report_contents(raw_dir)

    if args.skip_convert:
        print("Skipping conversion (--skip-convert). Run training/convert_to_yolo.py manually when ready.")
        return

    from training.convert_to_yolo import convert

    settings = get_settings()
    try:
        convert(raw_dir, processed_dir, settings.class_names)
    except NotImplementedError as exc:
        print(f"\nConversion could not proceed automatically: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Converted dataset ready at {processed_dir}")


if __name__ == "__main__":
    main()
