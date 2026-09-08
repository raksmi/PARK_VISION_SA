from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "parkvision_yolo26s" / "weights" / "best.pt"
SOURCE_DIR = BASE_DIR / "sample_images"
OUTPUT_DIR = BASE_DIR / "predictions"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    SOURCE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    images = [
        p for p in SOURCE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    if not images:
        print(f"Put a parking image inside: {SOURCE_DIR}")
        return

    model = YOLO(str(MODEL_PATH))

    model.predict(
        source=str(SOURCE_DIR),
        imgsz=512,
        conf=0.35,
        save=True,
        project=str(OUTPUT_DIR),
        name="annotated",
        exist_ok=True,
    )

    print(f"Annotated images saved in: {OUTPUT_DIR / 'annotated'}")

if __name__ == "__main__":
    main()
