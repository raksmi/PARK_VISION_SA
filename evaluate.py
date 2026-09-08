from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"
MODEL_PATH = BASE_DIR / "runs" / "parkvision_yolo26s" / "weights" / "best.pt"

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Missing dataset config: {DATA_YAML}. Run prepare_dataset.py first."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_PATH}. Run train.py first."
        )

    model = YOLO(str(MODEL_PATH))
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        plots=True,
    )

    lines = [
        "=== PARKVISION TEST RESULTS ===",
        f"mAP50-95: {metrics.box.map:.4f}",
        f"mAP50:    {metrics.box.map50:.4f}",
        f"mAP75:    {metrics.box.map75:.4f}",
        f"Precision:{metrics.box.mp:.4f}",
        f"Recall:   {metrics.box.mr:.4f}",
    ]
    print("\n" + "\n".join(lines))

    report_path = BASE_DIR / "evaluation_results.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {report_path} — copy these numbers into your README.")

if __name__ == "__main__":
    main()
