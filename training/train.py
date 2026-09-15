"""Fine-tune a pretrained YOLOv8 segmentation checkpoint on the car damage dataset.

Usage:
    python training/train.py
    python training/train.py --epochs 50 --device cuda:0
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/dataset.yaml", help="Path to Ultralytics data.yaml")
    parser.add_argument("--weights", default=settings.fallback_weights_path, help="Pretrained checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="Use -1 for Ultralytics auto-batch")
    parser.add_argument("--device", default=settings.device, help="e.g. cpu, cuda:0")
    # Ultralytics auto-prepends its runs_dir/<task>/ (task="segment" for a
    # -seg model) to a relative project value, so leave this empty by
    # default — it resolves to <runs_dir>/segment/<name>. Passing
    # "runs/segment" here would double up into runs/segment/runs/segment/.
    parser.add_argument("--project", default="")
    parser.add_argument("--name", default="car_damage_seg")
    parser.add_argument("--patience", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )

    # Read the actual save_dir Ultralytics used rather than reconstructing it
    # from args — its project/task path resolution has non-obvious rules.
    best_weights = model.trainer.save_dir / "weights" / "best.pt"
    settings = get_settings()
    settings.model_weights_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, settings.model_weights_path)
    print(f"Copied best weights to {settings.model_weights_path}")


if __name__ == "__main__":
    main()
