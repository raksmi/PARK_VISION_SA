"""
prepare_dataset.py
-------------------
Normalises a downloaded parking-lot object-detection dataset (e.g. a
Roboflow "train / valid / test" export of a PKLot-derived set) into a
clean, standard YOLO dataset that train.py / evaluate.py can consume,
and writes a data.yaml describing it.

Why this exists
----------------
Downloaded exports differ slightly in layout. Two shapes are common:

  A) split/images/*.jpg + split/labels/*.txt         (most Roboflow YOLO exports)
  B) split/*.jpg + split/*.txt in the same folder     (flat / classic YOLO layout)

This script auto-detects which shape each split (train/valid/test) is
in, links every split into a single predictable structure at
`dataset/<split>/images` and `dataset/<split>/labels`, carries over (or
infers) the class names, and writes `dataset/data.yaml`.

Usage
-----
1. Unzip your downloaded dataset into a folder named "raw dataset" next
   to this script, e.g.:

     ParkVision_AI/
       raw dataset/
         train/...
         valid/...   (or "val")
         test/...

2. Run:

     python prepare_dataset.py

3. It creates:

     ParkVision_AI/
       dataset/
         data.yaml
         train/images  train/labels
         valid/images  valid/labels
         test/images   test/labels

   train.py and evaluate.py read `dataset/data.yaml` by default.

Symlinks are used instead of copying so this works even with a large
dataset and doesn't duplicate gigabytes of images. If symlinks aren't
supported on your OS/filesystem, it automatically falls back to
copying the files.
"""

from pathlib import Path
import shutil
import sys

try:
    import yaml
except ImportError:
    print("Missing dependency. Run: pip install pyyaml")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw dataset"
OUT_DIR = BASE_DIR / "dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Roboflow / common split-name variants we should recognise.
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "validation": "valid",
    "val": "valid",
    "test": "test",
    "testing": "test",
}


def find_splits(raw_dir: Path) -> dict:
    """Return {'train': Path, 'valid': Path, 'test': Path} for whichever exist."""
    found = {}
    for child in raw_dir.iterdir():
        if not child.is_dir():
            continue
        key = SPLIT_ALIASES.get(child.name.strip().lower())
        if key:
            found[key] = child
    return found


def detect_shape(split_dir: Path):
    """
    Returns (images_dir, labels_dir) for a split folder, handling both
    the images/+labels/ subfolder shape and the flat shape.
    """
    images_sub = split_dir / "images"
    labels_sub = split_dir / "labels"
    if images_sub.is_dir():
        return images_sub, (labels_sub if labels_sub.is_dir() else None)

    # Flat shape: images and .txt labels live together in split_dir.
    has_images = any(p.suffix.lower() in IMAGE_EXTS for p in split_dir.iterdir())
    if has_images:
        return split_dir, split_dir

    raise FileNotFoundError(f"Could not find images inside {split_dir}")


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def build_split(split_name: str, split_dir: Path, out_dir: Path):
    images_dir, labels_dir = detect_shape(split_dir)

    out_images = out_dir / split_name / "images"
    out_labels = out_dir / split_name / "labels"

    n_images = 0
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        link_or_copy(img_path, out_images / img_path.name)
        n_images += 1

        if labels_dir is not None:
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                link_or_copy(label_path, out_labels / label_path.name)

    n_labels = len(list(out_labels.glob("*.txt"))) if out_labels.exists() else 0
    return n_images, n_labels


def find_class_names(raw_dir: Path, splits: dict) -> list:
    """Look for an existing data.yaml / classes.txt; else infer from label files."""
    for candidate in [raw_dir / "data.yaml", raw_dir / "data.yml"]:
        if candidate.exists():
            with open(candidate) as f:
                data = yaml.safe_load(f)
            names = data.get("names")
            if isinstance(names, dict):
                return [names[i] for i in sorted(names, key=int)]
            if isinstance(names, list):
                return names

    for candidate in [raw_dir / "classes.txt", raw_dir / "_darknet.labels"]:
        if candidate.exists():
            return [line.strip() for line in candidate.read_text().splitlines() if line.strip()]

    # Fall back: scan label files for the highest class index used.
    max_id = -1
    for split_dir in splits.values():
        _, labels_dir = detect_shape(split_dir)
        if labels_dir is None:
            continue
        for txt in labels_dir.glob("*.txt"):
            for line in txt.read_text().splitlines():
                parts = line.split()
                if parts:
                    max_id = max(max_id, int(parts[0]))

    if max_id < 0:
        raise RuntimeError(
            "Could not find class names anywhere and no label files to infer "
            "from. Add a data.yaml or classes.txt to 'raw dataset/'."
        )

    if max_id == 1:
        # Most likely binary occupied/empty slot detection.
        return ["occupied", "empty"]
    return [f"class{i}" for i in range(max_id + 1)]


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"'{RAW_DIR}' not found. Put your unzipped dataset (with "
            f"train/valid/test folders) inside a folder named 'raw dataset' "
            f"next to this script."
        )

    splits = find_splits(RAW_DIR)
    if "train" not in splits:
        raise FileNotFoundError(
            f"No 'train' split found inside {RAW_DIR}. Found: {list(splits)}"
        )

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    names = find_class_names(RAW_DIR, splits)
    print(f"Classes detected: {names}")

    summary = {}
    for split_name, split_dir in splits.items():
        n_images, n_labels = build_split(split_name, split_dir, OUT_DIR)
        summary[split_name] = (n_images, n_labels)
        print(f"{split_name:5s}: {n_images:5d} images, {n_labels:5d} label files")

    data_yaml = {
        "path": str(OUT_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images" if "valid" in splits else "train/images",
        "names": {i: n for i, n in enumerate(names)},
    }
    if "test" in splits:
        data_yaml["test"] = "test/images"

    yaml_path = OUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"\nWrote {yaml_path}")
    if "valid" not in splits:
        print(
            "WARNING: no 'valid'/'val' split found — using the train split for "
            "validation too. Re-split your data 70/15/15 if possible."
        )
    total_images = sum(n for n, _ in summary.values())
    print(f"Total images across all splits: {total_images}")
    print("\nDataset ready. Next: python train.py")


if __name__ == "__main__":
    main()
