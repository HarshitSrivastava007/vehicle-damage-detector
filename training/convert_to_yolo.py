"""Normalize a raw downloaded dataset into Ultralytics YOLO-seg layout:

    processed_dir/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt   (one line per instance: "class_id x1 y1 x2 y2 ... xn yn", normalized 0-1)

Verified against the official CarDD dataset (Kaggle: issamjebnouni/cardd),
which ships train.json/val.json/test.json in real COCO format with polygon
segmentation (not RLE, not bounding-box-only). The `named_splits` path in
convert_coco() uses that dataset's own curated split rather than inventing
one. The single-json fallback path remains for other COCO-shaped datasets
that don't ship pre-split annotation files. If segmentation masks turn up
as COCO RLE rather than polygon lists, decoding them will need
`pycocotools` added as a dependency.
"""

import json
import random
import shutil
from pathlib import Path
from typing import Literal

# Our class list merges the source dataset's separate broken-part categories
# (e.g. CarDD's "glass_shatter" and "lamp_broken") into one "broken" class.
# Keys are matched after normalize_category_name() (lowercased, spaces ->
# underscores), so only underscore-form keys are needed here.
CATEGORY_ALIASES = {
    "glass_shatter": "broken",
    "lamp_broken": "broken",
    "broken_lamp": "broken",
    "broken_glass": "broken",
}


def normalize_category_name(raw_name: str) -> str:
    return raw_name.strip().lower().replace(" ", "_")


def detect_raw_format(raw_dir: Path) -> Literal["yolo", "coco", "unknown"]:
    if any(raw_dir.rglob("labels/*.txt")) or list(raw_dir.rglob("*.txt")):
        images_present = any(raw_dir.rglob("*.jpg")) or any(raw_dir.rglob("*.png"))
        labels_present = any(raw_dir.rglob("*.txt"))
        if images_present and labels_present:
            return "yolo"

    for json_path in raw_dir.rglob("*.json"):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys():
            return "coco"

    return "unknown"


def train_val_split(
    pairs: list[tuple[Path, Path]], val_ratio: float = 0.15, seed: int = 42
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    split_index = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_index], shuffled[split_index:]


def _write_split(pairs: list[tuple[Path, Path]], images_out: Path, labels_out: Path) -> None:
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for image_path, label_path in pairs:
        shutil.copy2(image_path, images_out / image_path.name)
        shutil.copy2(label_path, labels_out / label_path.name)


def convert_already_yolo(raw_dir: Path, processed_dir: Path) -> None:
    image_paths = [p for p in raw_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    pairs: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        label_path = image_path.with_suffix(".txt")
        if label_path.exists():
            pairs.append((image_path, label_path))

    if not pairs:
        raise NotImplementedError(
            "detected YOLO-like format (found .txt label files) but could not "
            "pair any image with a matching label file — inspect data/raw/ "
            "and adjust convert_to_yolo.convert_already_yolo()."
        )

    train_pairs, val_pairs = train_val_split(pairs)
    _write_split(train_pairs, processed_dir / "images" / "train", processed_dir / "labels" / "train")
    _write_split(val_pairs, processed_dir / "images" / "val", processed_dir / "labels" / "val")


def convert_coco_to_yolo_seg(
    coco_json_path: Path,
    images_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    class_name_map: dict[int, str],
) -> None:
    with open(coco_json_path) as f:
        coco = json.load(f)

    name_to_id = {name: idx for idx, name in class_name_map.items()}
    category_id_to_class_id = {}
    for category in coco["categories"]:
        raw_name = category["name"]
        normalized = normalize_category_name(raw_name)
        canonical_name = CATEGORY_ALIASES.get(normalized, normalized)
        class_id = name_to_id.get(canonical_name)
        if class_id is None:
            raise NotImplementedError(
                f"COCO category {raw_name!r} (normalized: {canonical_name!r}) has no "
                "match in the configured class list (configs/dataset.yaml) and no "
                "entry in CATEGORY_ALIASES — add one or update the class list."
            )
        category_id_to_class_id[category["id"]] = class_id

    images_by_id = {image["id"]: image for image in coco["images"]}
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_id, image_info in images_by_id.items():
        annotations = annotations_by_image.get(image_id, [])
        if not annotations:
            continue

        width, height = image_info["width"], image_info["height"]
        lines = []
        for annotation in annotations:
            segmentation = annotation.get("segmentation")
            if not segmentation or isinstance(segmentation, dict):
                # dict form is RLE-encoded — not handled here, see module docstring.
                raise NotImplementedError(
                    f"annotation {annotation.get('id')} has no polygon segmentation "
                    "(possibly RLE-encoded) — add pycocotools-based RLE decoding to "
                    "convert_coco_to_yolo_seg() if this dataset uses RLE masks."
                )

            class_id = category_id_to_class_id[annotation["category_id"]]
            polygon = segmentation[0]
            normalized = [
                coord / width if idx % 2 == 0 else coord / height
                for idx, coord in enumerate(polygon)
            ]
            lines.append(" ".join([str(class_id)] + [f"{v:.6f}" for v in normalized]))

        image_path = images_dir / image_info["file_name"]
        if not image_path.exists():
            continue

        shutil.copy2(image_path, out_images_dir / image_path.name)
        label_path = out_labels_dir / (image_path.stem + ".txt")
        label_path.write_text("\n".join(lines) + "\n")


def _find_images_dir(raw_dir: Path, coco_json_path: Path, split_name: str) -> Path:
    """Images for a split's json usually live in a same-named sibling folder
    (e.g. raw_dir/train.json + raw_dir/train/*.jpg), sometimes alongside the
    json itself. Try the common layouts rather than assuming one."""
    candidates = [raw_dir / split_name, coco_json_path.parent / split_name, coco_json_path.parent]
    for candidate in candidates:
        if candidate.is_dir() and (any(candidate.glob("*.jpg")) or any(candidate.glob("*.png"))):
            return candidate
    raise NotImplementedError(
        f"could not locate an image directory for {coco_json_path.name} — checked "
        f"{[str(c) for c in candidates]}. Inspect data/raw/ and adjust _find_images_dir()."
    )


def convert_coco(raw_dir: Path, processed_dir: Path, class_name_map: dict[int, str]) -> None:
    # Prefer a dataset's own pre-made train/val/test split (named json files)
    # over inventing our own random split, so we don't discard curated splits.
    named_splits = {
        split: raw_dir / f"{split}.json"
        for split in ("train", "val", "test")
        if (raw_dir / f"{split}.json").exists() and _is_coco_json(raw_dir / f"{split}.json")
    }

    if named_splits:
        for split, coco_json_path in named_splits.items():
            images_dir = _find_images_dir(raw_dir, coco_json_path, split)
            convert_coco_to_yolo_seg(
                coco_json_path,
                images_dir,
                processed_dir / "images" / split,
                processed_dir / "labels" / split,
                class_name_map,
            )
        return

    json_candidates = [p for p in raw_dir.rglob("*.json") if _is_coco_json(p)]
    if not json_candidates:
        raise NotImplementedError("expected a COCO-format *.json but none was found under data/raw/")

    coco_json_path = json_candidates[0]
    images_dir = coco_json_path.parent

    all_pairs_dir = processed_dir / "_all"
    convert_coco_to_yolo_seg(
        coco_json_path,
        images_dir,
        all_pairs_dir / "images",
        all_pairs_dir / "labels",
        class_name_map,
    )

    image_paths = sorted((all_pairs_dir / "images").glob("*"))
    pairs = [(p, (all_pairs_dir / "labels" / (p.stem + ".txt"))) for p in image_paths]
    pairs = [(img, lbl) for img, lbl in pairs if lbl.exists()]

    train_pairs, val_pairs = train_val_split(pairs)
    _write_split(train_pairs, processed_dir / "images" / "train", processed_dir / "labels" / "train")
    _write_split(val_pairs, processed_dir / "images" / "val", processed_dir / "labels" / "val")
    shutil.rmtree(all_pairs_dir, ignore_errors=True)


def _is_coco_json(path: Path) -> bool:
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys()


def convert(raw_dir: Path, processed_dir: Path, class_name_map: dict[int, str]) -> None:
    raw_format = detect_raw_format(raw_dir)

    if raw_format == "yolo":
        convert_already_yolo(raw_dir, processed_dir)
    elif raw_format == "coco":
        convert_coco(raw_dir, processed_dir, class_name_map)
    else:
        raise NotImplementedError(
            f"raw dataset format at {raw_dir} was not recognized as YOLO or COCO — "
            "inspect data/raw/ manually and either restructure it to match one of "
            "those layouts or extend training/convert_to_yolo.py's detect_raw_format()."
        )
