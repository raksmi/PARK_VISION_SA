# 🅿️ ParkVision AI

## Intelligent Urban Parking Analytics & Space Optimisation Platform

ParkVision AI is a computer-vision system built for **UrbanFlow AI Pvt. Ltd.** that analyses parking-lot images, detects individual parking slots, classifies each as **occupied** or **empty**, and converts those detections into real-time parking-availability insights and recommendations.

**Live app:** `<PASTE YOUR STREAMLIT CLOUD LINK HERE>`
**GitHub repo:** `<PASTE YOUR GITHUB REPO LINK HERE>`

---

## 1. Problem & Requirements

Urban drivers frequently waste time, fuel, and patience searching for a free parking spot, which adds to city-wide congestion. This project treats parking-occupancy detection as a **slot-level object detection problem**, not a single whole-image classification problem, because a parking lot image typically contains many slots that must each be judged independently.

| | |
|---|---|
| **Input** | A parking-lot image (JPG/PNG) |
| **Output** | Slot-level bounding boxes, each labelled `occupied` or `empty`, plus lot-level availability metrics |
| **Approach chosen** | Full-image object detection with YOLO (rather than crop-then-classify), because it localises every slot in a single forward pass and scales to lots with irregular, non-fixed layouts |

**Real-world challenges considered:** shadows and changing lighting, partial occlusion of cars, weather variation (sunny/cloudy/rainy), and camera angle differences between parking lots.

---

## 2. Dataset & Preprocessing

- **Source:** a PKLot-derived parking-slot dataset (occupied vs. empty slot annotations), downloaded and used locally per the assessment brief.
- **Subset size:** at least 100 images per class (occupied / empty), as required.
- **Image size:** standardised to 640×640 for the detection model.
- **Augmentation:** rotation, horizontal flip, and brightness adjustment applied to improve robustness to lighting/weather/angle variation. *(Ultralytics applies its default augmentation pipeline — mosaic, HSV colour jitter, flip — during `train.py`; document here if you additionally pre-augmented the raw images before training.)*
- **Split:** 70% train / 15% validation / 15% test.
- **Structure:** the raw download (train/valid/test folders) is normalised by [`prepare_dataset.py`](prepare_dataset.py) into:

```
dataset/
├── data.yaml
├── train/images  train/labels
├── valid/images  valid/labels
└── test/images   test/labels
```

Run once before training:

```bash
python prepare_dataset.py
```

It auto-detects whichever folder layout the download came in, links images/labels into the structure above, infers or copies the class names (`occupied`, `empty`), and prints per-split counts so the balance across classes can be checked before training.

---

## 3. Model

| | |
|---|---|
| Architecture | Ultralytics **YOLO26s** |
| Task | Object detection (2 classes: `occupied`, `empty`) |
| Input size | 640 × 640 |
| Epochs | 50 max, early stopping (`patience=15`) |
| Batch size | 8 |
| Pretrained weights | COCO-pretrained `yolo26s.pt`, fine-tuned on the parking dataset |

Train:

```bash
python train.py
```

Weights are written to `runs/parkvision_yolo26s/weights/best.pt`.

### Evaluation

```bash
python evaluate.py
```

Reports on the held-out **test** split and saves the numbers to `evaluation_results.txt`:

| Metric | Value |
|---|---|
| mAP50-95 | `<FILL IN AFTER RUNNING evaluate.py>` |
| mAP50 | `<FILL IN>` |
| mAP75 | `<FILL IN>` |
| Precision | `<FILL IN>` |
| Recall | `<FILL IN>` |

*(Add a short paragraph here once you have the numbers: how the model performed, which errors were common — e.g. missed detections in heavy shadow, false positives on partially-visible cars — and what you tried to improve it, e.g. more data, fixed labels, tuned confidence threshold.)*

### Testing

Automated unit tests for the insight logic:

```bash
python -m unittest test_parking_logic.py -v
```

Manual robustness testing on unseen images (different lighting/weather/angle, not used in training):

```bash
python predict.py
```
(reads images from `sample_images/`, writes annotated results to `predictions/annotated/`)

---

## 4. Parking Insight Logic

Implemented in [`parking_logic.py`](parking_logic.py). From the detections for one image, the app computes:

- Total parking slots = occupied + empty
- Occupied slots, available slots
- Occupancy % = occupied / total × 100
- Congestion level:

| Occupancy | Status |
|---|---|
| < 40% | 🟢 Low |
| 40–75% | 🟡 Moderate |
| > 75% | 🔴 High |

- A plain-language recommendation ("proceed to park" vs. "try another area")

---

## 5. Web App (Streamlit)

```bash
streamlit run app.py
```

The dashboard lets a user upload a parking-lot image and shows, side by side:

- The image annotated with colour-coded boxes (green = empty, red = occupied)
- Total / occupied / available slot counts and occupancy %
- A congestion badge and a recommendation

*(Add 2–3 screenshots of the running app here once deployed, e.g.)*

```
![Upload screen](screenshots/upload.png)
![Annotated result + insights](screenshots/results.png)
```

---

## 6. Deployment

1. Push this repository to GitHub (see below).
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud), sign in with GitHub, select this repo, set the main file to `app.py`.
3. Streamlit Cloud installs `requirements.txt` and builds the app; note the trained weights (`runs/parkvision_yolo26s/weights/best.pt`) must be committed (or fetched at startup) for the app to load a model on the cloud.
4. Test the deployed public link with a fresh image before submitting.

---

## 7. Project Structure

```text
ParkVision_AI/
├── app.py                 # Streamlit dashboard
├── train.py                # trains YOLO26s on dataset/
├── evaluate.py              # test-set metrics -> evaluation_results.txt
├── predict.py               # batch inference on sample_images/
├── parking_logic.py         # occupancy/congestion/recommendation logic
├── prepare_dataset.py       # normalises raw download into dataset/
├── test_parking_logic.py    # unit tests
├── requirements.txt
├── README.md
├── raw dataset/             # your downloaded dataset (not tracked by default)
├── dataset/                 # normalised dataset (built by prepare_dataset.py)
├── runs/                    # training outputs incl. best.pt
├── sample_images/
└── predictions/
```

---

## 8. References

- Deep Learning Based Smart Parking Occupancy Detection using Computer Vision
- Vision-Based Parking Slot Detection using Deep Learning
- PKLot: A Robust Dataset for Parking Lot Classification
- Ultralytics YOLO Object Detection Documentation
- Real-Time Parking Occupancy Detection using CNN and Computer Vision
- LearnOpenCV — Object Detection using YOLO
- Streamlit Official Documentation
- OpenCV Official Documentation

*(These are the reference materials provided in the assignment brief. Add the specific paper titles/links you actually read and cite anything else you used, e.g. dataset card, Roboflow project page.)*

---

## Submission checklist

- [ ] GitHub repo access given to `ai.assignments@wacpinternational.org`
- [ ] Code (.py) uploaded
- [ ] Dataset uploaded (or a representative subset + a note on where to get the rest, if the full dataset is too large for GitHub)
- [ ] Trained model weights uploaded or reproducible via `train.py`
- [ ] Live Streamlit Cloud link works
- [ ] README has findings, references, data-prep description, model details, metrics, and screenshots
- [ ] Submission PDF includes GitHub link, name, registration number, CRS name, course name, school name
