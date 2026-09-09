# Licence Number Plate Detection

A Python-based repository for detecting and reading vehicle licence (license) plates from images and video. This README explains the core concepts, the end-to-end workflow, recommended architectures, evaluation metrics, numerical-analysis templates and examples, and practical tips for training and deployment.

Table of contents
- Project overview
- Key concepts (short)
- End-to-end workflow
- Installation & quick start
- Directory structure (recommended)
- Models & algorithms (options & trade-offs)
- Data: collection, annotation & augmentation
- Training: hyperparameters & loss functions
- Inference & post-processing
- Evaluation & numerical analysis (metrics, worked examples)
- Experiments & ablation study ideas
- Optimization & deployment
- Common pitfalls & troubleshooting
- References and further reading
- Contribution & license

Project overview
This project aims to detect licence plates in images/video and recognize the characters (OCR) on the plate. Typical outputs:
- Bounding box (x, y, w, h) for each plate instance.
- Cropped, rectified plate image per detection.
- Recognized text string for plate characters.

Use-cases include traffic monitoring, parking systems, security checkpoints and automatic tolling.

Key concepts
- Object detection: localize plates with bounding boxes. One-stage detectors (YOLO, SSD) trade some accuracy for speed; two-stage detectors (Faster R-CNN) usually give better localization but are slower.
- OCR (Optical Character Recognition): read characters after localization. Options: CRNN, Tesseract, transformer-based OCR models.
- Preprocessing: image resizing, histogram equalization, denoising, adaptive thresholding for OCR.
- Post-processing: Non-Maximum Suppression (NMS), perspective correction, morphological ops and character-level confidence filtering.
- Metrics: IoU, precision, recall, F1-score, mean Average Precision (mAP), character accuracy.

End-to-end workflow
1. Data collection: images and video frames representing target environments (angles, lighting, plate types).
2. Annotation: bounding boxes for plates + transcription for OCR (text labels). Use COCO/VOC or YOLO format.
3. Preprocessing & augmentation: normalize, random crop/scale/rotate, blur, brightness variation, synthetic plate generation.
4. Detector training: choose architecture, train to predict plate boxes and confidences.
5. Plate rectification: apply homography/perspective transform on detected boxes for better OCR input.
6. OCR model training: train on cropped plate images and transcriptions (CTC/attention/transformer-based).
7. Post-processing & heuristics: country-specific character patterns, reject improbable outputs.
8. Evaluation: compute detection metrics and OCR accuracy, report combined end-to-end accuracy.
9. Deployment: optimize model (pruning/quantization), package inference pipeline for CPU/GPU/edge.

Installation & quick start
Prerequisites
- Python 3.8+
- pip, virtualenv (recommended)
- CUDA (optional for GPU)

Example install (Unix-like):
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Quick inference (example CLI usage; adapt to repo scripts)
```bash
# detect and OCR on an image
python scripts/infer.py --weights weights/detector.pt --ocr weights/ocr.pth --input images/vehicle.jpg --output results/
```

Directory structure (recommended)
- data/
  - raw/                  # raw images and video
  - annotations/          # bounding boxes + transcriptions (COCO/YOLO)
  - processed/            # preprocessed images / train/val splits
- notebooks/              # EDA and experiments
- models/
  - detector/             # detector model definitions
  - ocr/                  # OCR model definitions
- scripts/
  - train_detector.py
  - train_ocr.py
  - infer.py
  - evaluate.py
- utils/                  # preprocessing, augmentations, metrics
- requirements.txt
- README.md

Models & algorithms (options & trade-offs)
1. Classical vision + ML
   - Haar cascades / HOG+SVM
   - Pros: very fast, simple to set up.
   - Cons: fragile under variation (angle, plate designs).

2. One-stage deep detectors
   - YOLOv5/YOLOv8, SSD
   - Pros: real-time inference (high FPS), single network.
   - Cons: slightly lower localization accuracy on small or heavily occluded plates.

3. Two-stage detectors
   - Faster R-CNN, Mask R-CNN
   - Pros: higher precision for small objects, better recall in clutter.
   - Cons: slower inference, heavier compute.

4. OCR choices
   - Tesseract: quick baseline, limited robustness out-of-the-box for stylized plates.
   - CRNN (CNN + RNN + CTC): widely used for sequence recognition on constrained character sets.
   - Transformer-based sequence models: state-of-the-art for complex scripts.

Data: collection, annotation & augmentation
- Aim for diversity: viewpoints, distances, lighting, occlusion, dirty/worn plates.
- Annotation fields:
  - Bounding box coordinates (x_min, y_min, x_max, y_max)
  - Transcription string (plate text)
  - Optionally plate type (front/rear), country, plate color
- Augmentation suggestions:
  - Photometric: brightness, contrast, hue, blur, noise
  - Geometric: scaling, rotation (small angles), perspective transform
  - Synthetic plates: generate plates with various fonts/backgrounds to increase OCR robustness
- Splits: typical 70/15/15 for train/val/test; ensure no vehicle appears in both train and test if possible.

Training: hyperparameters & loss functions
Detector
- Loss: combination of localization loss (IoU-based or L1/L2), objectness (binary cross-entropy), and classification (if multi-class).
- Typical hyperparameters (example for YOLO-like):
  - batch_size: 16–64 (depending on GPU)
  - lr: 0.001 (with warmup and cosine or step decay)
  - momentum: 0.9
  - weight_decay: 5e-4
  - epochs: 50–300 (depends on dataset size)
- Anchors: tune for plate aspect ratios; plates are often wide rectangles (e.g., 4:1–2.5:1).

OCR
- Loss: Connectionist Temporal Classification (CTC) for CRNN; cross-entropy with teacher forcing for attention models.
- Typical hyperparameters:
  - lr: 0.0001–0.001
  - batch_size: 32–256 (depending on GPU)
  - epochs: 30–200

Inference & post-processing
- Confidence threshold: typical 0.3–0.5 for initial filtering.
- Non-maximum suppression (NMS): IoU threshold 0.4–0.6 to suppress duplicate detections.
- Plate rectification:
  - Expand box slightly, detect plate corners with contour analysis or use predicted quadrilateral if model supports it.
  - Apply perspective transform to get frontal crop of plate for OCR.
- OCR confidence / language validation:
  - Apply regex or country-specific format checks to reject invalid outputs.
  - Use edit distance to a whitelist of allowed sequences (where applicable).

Evaluation & numerical analysis
Detection metrics
- Intersection over Union (IoU)
  IoU = area(B_pred ∩ B_gt) / area(B_pred ∪ B_gt)
- True positive (TP): predicted box matches GT with IoU >= threshold (commonly 0.5).
- False positive (FP): predicted box not matched to any GT.
- False negative (FN): GT not detected.
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
- mean Average Precision (mAP): area under precision-recall curve, averaged across classes (single-class plate detection -> AP).

OCR metrics
- Character accuracy = (correct_chars) / (total_chars)
- Sequence (plate) accuracy = fraction of full plate strings exactly matched.
- Normalized edit distance (NED) = 1 - (edit_distance / max_len)

Combined end-to-end metric
- End-to-end accuracy = fraction of GT plates for which:
  - detection IoU >= IoU_thresh, AND
  - OCR transcription matches GT (exact or within acceptable edit distance)

Worked numerical example (hypothetical)
Suppose on test set with 100 ground-truth plates:
- Predictions: 110 boxes (some duplicates)
- After matching with IoU>=0.5:
  - TP = 88
  - FP = 22 (110 - 88)
  - FN = 12 (100 - 88)

Compute detection metrics:
- Precision = 88 / (88 + 22) = 0.8 (80%)
- Recall = 88 / (88 + 12) = 0.88 (88%)
- F1 = 2 * 0.8 * 0.88 / (0.8 + 0.88) ≈ 0.839 (83.9%)

OCR results for the 88 TP detections:
- Exact plate matches (full): 70
- Average per-character accuracy across these plates: 95%

End-to-end metrics:
- End-to-end exact-match accuracy = 70 / 100 = 70%
- Combined idea: consider TP only when both detection and OCR succeed; otherwise count as failure.

How to compute mAP (brief)
1. Rank predictions by confidence.
2. For each confidence threshold, compute precision and recall.
3. Plot precision vs recall, compute area under curve (AP).
4. mAP is mean(APs) across classes (one class => mAP = AP).

Example benchmark table (illustrative)
| Model | mAP@0.5 | Detection Precision | Detection Recall | OCR Seq Acc | FPS (GPU) |
|-------|---------|---------------------|------------------|-------------|-----------|
| Haar cascade (classical) | 0.25 | 0.35 | 0.55 | 0.45 | 120 |
| YOLOv5s (one-stage) | 0.72 | 0.78 | 0.84 | 0.78 | 45 |
| Faster R-CNN (two-stage) | 0.80 | 0.83 | 0.86 | 0.82 | 8 |
| YOLOv8 (trained + rectification) | 0.78 | 0.81 | 0.85 | 0.86 | 55 |

Notes: numbers above are illustrative. Use your dataset to compute real metrics; report mAP at IoU thresholds (0.5, 0.75) for a more complete view.

Experiments & ablation study ideas
- Effect of perspective correction: train and test with/without rectification and measure OCR accuracy change.
- Anchor tuning vs. anchor-free: evaluate plate localization improvements.
- Synthetic data augmentation: measure gain from adding synthetic plates.
- OCR model architecture: CRNN vs transformerOCR.
- Confidence threshold and NMS: sweep thresholds and plot precision/recall curves.
- Real-time vs accuracy trade-off: compare different model sizes (tiny vs large).

Optimization & deployment
- Quantize to INT8 (TensorRT, ONNX Runtime) for CPU/edge deployment.
- Prune or use smaller backbones (MobileNet, EfficientNet-lite).
- Batch inference for throughput when processing recordings.
- Use pipelined inference: detector -> rectification -> OCR with parallel processing to maximize FPS.

Common pitfalls & troubleshooting
- Low recall for small plates: increase input resolution or use multi-scale training; tune anchors.
- OCR fails due to motion blur: add motion blur augmentation and temporal averaging over video frames.
- Wrong plate format predictions: apply country-specific regex filters and language models.
- Overfitting: use validation set, early stopping, stronger augmentation.
- Duplicate/bad detections: adjust NMS IoU and score thresholds.

Practical tips
- Normalize image sizes consistently between training and inference.
- Use mixed precision training (fp16) to speed up training with big models on modern GPUs.
- Keep a held-out test set from a different source to estimate generalization.
- For video, perform temporal smoothing of results to reduce flicker.

References and further reading
- YOLO family papers and repositories
- Faster R-CNN original paper
- CRNN: "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"
- Tesseract OCR
- Papers on license plate recognition (ALPR) and synthetic plate generation

Contribution & license
- Contributions are welcome via issues and pull requests. Create reproducible experiments (seed, dataset split, hyperparameters).
- Add a license file to the repo (e.g., MIT) if you want to allow reuse.

Appendix — quick formulas and command snippets
Detection metrics:
```text
IoU = intersection_area / union_area
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Example evaluation command (adapt to repo):
```bash
python scripts/evaluate.py --pred results/predictions.json --gt data/annotations/test.json --iou-thres 0.5
```

How to report numerical analysis in papers or README
- Always state dataset size, splits, exact IoU thresholds, and whether mAP is averaged across IoU thresholds (COCO mAP vs PASCAL mAP).
- Report per-category (if multiple) metrics, and compute 95% confidence intervals (via bootstrapping) for robustness.
- Provide confusion matrices for OCR character-level errors and example failure cases.

If you want, I can:
- Generate a ready-to-use evaluation notebook (with metric calculations and plots) for this repo.
- Create example train and inference scripts (skeleton) tailored to your existing codebase.
- Draft a small experiment plan (hyperparameter grid + expected compute/time).
