import os
import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRMedium

# ── Your 17 classes in alphabetical order
# RF-DETR assigns 0-based IDs sorted alphabetically at training time
CLASSES = [
    "AmurTiger",   # 0
    "Badger",      # 1
    "BlackBear",   # 2
    "Cow",         # 3
    "Dog",         # 4
    "Hare",        # 5
    "Leopard",     # 6
    "LeopardCat",  # 7
    "MuskDeer",    # 8
    "RaccoonDog",  # 9
    "RedFox",      # 10
    "RoeDeer",     # 11
    "Sable",       # 12
    "SikaDeer",    # 13
    "Weasel",      # 14
    "WildBoar",    # 15
    "Y-T-Marten",  # 16
]

WEIGHTS   = os.path.abspath("weights.pt")
IMAGE     = "images (8).jpg"
THRESHOLD = 0.3

# ── Load model with your fine-tuned weights
print("Loading model...")
model = RFDETRMedium(pretrain_weights=WEIGHTS)
print("Model loaded. Running prediction...")

# ── Run prediction
image = Image.open(IMAGE)
detections = model.predict(image, threshold=THRESHOLD)

print(f"\nDetections (threshold={THRESHOLD}):")
print(f"  Total: {len(detections)}")

if len(detections) == 0:
    print("  No detections above threshold.")
else:
    for i in range(len(detections)):
        cid   = detections.class_id[i]
        score = detections.confidence[i]
        box   = detections.xyxy[i]  # x1, y1, x2, y2 in pixels
        name  = CLASSES[cid] if cid < len(CLASSES) else f"class_{cid}"
        print(f"  [{i}] class_id={cid}  name={name}  confidence={score:.3f}  box={box.astype(int).tolist()}")

# ── Draw and save result image
img_cv = cv2.imread(IMAGE)
h, w   = img_cv.shape[:2]

for i in range(len(detections)):
    cid   = detections.class_id[i]
    score = detections.confidence[i]
    x1, y1, x2, y2 = detections.xyxy[i].astype(int)
    name  = CLASSES[cid] if cid < len(CLASSES) else f"class_{cid}"

    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{name} {score:.2f}"
    cv2.putText(img_cv, label, (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

out_path = "prediction_result.jpg"
cv2.imwrite(out_path, img_cv)
print(f"\nAnnotated image saved to: {out_path}")
