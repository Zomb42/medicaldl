# YOLOv8 Medical Imaging Demo

This Streamlit app demonstrates YOLOv8-based medical image classification, object detection, and segmentation workflows. It includes demo images and previously trained weights so the app can be tried without downloading the full training datasets.

> This project is for education and experimentation only. It is not a medical device and must not be used for diagnosis or clinical decision-making.

## Project Background

This repository began as a modified version of [sevdaimany/YOLOv8-Medical-Imaging](https://github.com/sevdaimany/YOLOv8-Medical-Imaging). The app was adapted for EE-24235-01, The Science and Engineering of Digital Photography, with an added musculoskeletal fracture classifier trained on FracAtlas-style X-ray images.

The app currently exposes four demo tasks:

- Musculoskeletal fracture classification
- Blood cell object detection
- Lung X-ray classification
- Breast ultrasound segmentation

## Quick Start

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate yolomedical
streamlit run app.py
```

If you already have the `yolomedical` environment, update it instead:

```bash
conda activate yolomedical
pip install --upgrade -r requirements.txt
streamlit run app.py
```

### Option 2: Python venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Repository Layout

- `app.py`: Streamlit entry point
- `classification/`: Lung X-ray classification code
- `detection/`: Blood cell detection code
- `segmentation/`: Ultrasound segmentation code
- `muscledetect/`: Musculoskeletal fracture classification code
- `DEMO_IMAGES/`: Images used when no file is uploaded
- `models/`: Small trained weights used by the demo app
- `runs/`: Generated training outputs, ignored by git

The full training datasets and generated run artifacts are intentionally not part of the public source tree. Keep local datasets under `data/` or `FracAtlas/`; both paths are ignored by git.

## Security Notes

The dependency pins were refreshed to address Dependabot alerts for vulnerable `Pillow` and `streamlit` releases. Keep dependencies current with:

```bash
pip install --upgrade -r requirements.txt
```

For public deployments, avoid accepting untrusted files beyond image uploads, run Streamlit behind normal access controls, and do not expose training data containing protected health information.

Dependabot intentionally ignores major-version updates for `numpy` and `opencv-python` while this project uses Python 3.10. OpenCV 5 requires NumPy 2+, and the latest NumPy releases have moved past Python 3.10 support, so those upgrades should be handled together with a future Python 3.12+ migration.

## Smoke Test

Run the lightweight import and model-file check with:

```bash
python -m unittest discover -s tests
```

## Training

Training hooks are present in `train_models()` inside `app.py`, but they assume the original datasets exist locally. To retrain models, update the dataset paths in the relevant module first:

- `classification/classify.py`
- `detection/detect.py`
- `segmentation/segment.py`
- `muscledetect/muscledetect.py`

Then uncomment the `train_models()` call at the bottom of `app.py`.

After retraining, copy the best weights you want the app to use into `models/`:

```bash
cp runs/detect/train/weights/best.pt models/blood-cell-detection.pt
cp runs/classify/train/weights/best.pt models/lung-classification.pt
cp runs/segment/train/weights/best.pt models/ultrasound-segmentation.pt
cp runs/muscleclassify/train/weights/best.pt models/fracture-classification.pt
```
