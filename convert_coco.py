from pathlib import Path
import json

ROOT = Path("raw dataset")

for split in ["train", "valid", "test"]:
    folder = ROOT / split
    ann_file = folder / "_annotations.coco.json"
    if not ann_file.exists():
        print(f"Skipping {split}: annotation file not found")
        continue

    data = json.loads(ann_file.read_text(encoding="utf-8"))
    images = {x["id"]: x for x in data["images"]}
    cats = {x["id"]: i for i, x in enumerate(sorted(data["categories"], key=lambda c: c["id"]))}
    anns = {}
    for a in data["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    count = 0
    for image_id, info in images.items():
        img = folder / info["file_name"]
        if not img.exists():
            continue
        label = folder / (Path(info["file_name"]).stem + ".txt")
        W, H = info["width"], info["height"]
        lines = []
        for a in anns.get(image_id, []):
            x, y, w, h = a["bbox"]
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            lines.append(f"{cats[a["category_id"]]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        label.write_text("\\n".join(lines), encoding="utf-8")
        count += 1
    print(f"{split}: created {count} YOLO label files")

print("COCO - conversion complete.")
