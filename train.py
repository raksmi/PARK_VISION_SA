from pathlib import Path
from ultralytics import YOLO
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"
RUNS_DIR = BASE_DIR / "runs"

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_YAML}. "
            "Run 'python prepare_dataset.py' first to build the dataset "
            "from 'raw dataset/'."
        )

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("yolo26n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=15,
        imgsz=512,
        batch=4,
        patience=5,
        device=device,
        project=str(RUNS_DIR),
        name="parkvision_yolo26s",
        exist_ok=True,
        plots=True,
        pretrained=True,
    )

if __name__ == "__main__":
    main()
