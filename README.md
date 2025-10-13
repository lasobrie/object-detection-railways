# Railway Track Fault Classification (Ultralytics YOLO)

Lightweight prototype to **classify** railway track images as **OK** or **FAIL** using Ultralytics (YOLO*) with PyTorch. The notebook sets up a simple project folder, and shows how to start training a YOLO **classification** model.

*Khanam, R., & Hussain, M. (2024). Yolov11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725.

## Contents
- `Object_Detection_Vibrant_Virginia.ipynb` -> main notebook
- `requirements.txt` -> Python dependencies
- `notebooks/` -> contains the object classification notebook, and the video-to-images notebook
- `data/` -> sample images from our project, for larger image sets use Kaggle
- `model/` -> the starting YOLO 11 model for classification: yolo11n-cls.pt
- `runs/` -> the train logs of the updated YOLO model for railway tracks breaks and failures
- `videos/` -> the videos of the model trains with no defects and defects, used to extract images


## Quickstart
```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Launch Jupyter and open the notebook
jupyter lab
# or: jupyter notebook
```

> **GPU note:** Training benefits from an NVIDIA GPU + CUDA. CPU-only works for small tests but will be slow. If you install `torch` manually for your platform/CUDA, do it **before** installing `ultralytics` or pin the appropriate wheel from https://pytorch.org.

## Dataset
This notebook uses images from the Virginia Tech Systems Performance Lab. An alternative may be to use open-source data which can be downloaded via Kaggle. For example:
- **salmaneunus/railway-track-fault-detection** (via `kagglehub`)

At runtime it copies the dataset into a local folder:
```
railway-track-fault-detection/
```

## Training (classification)
Inside the notebook, look for the **Train (classification task)** cell. If it’s commented, you can start with defaults like:
```python
from ultralytics import YOLO

# Create a classification model (e.g., yolov8n-cls) and train
model = YOLO("yolov11n-cls.pt")  # or "yolov10n-cls.pt" if available
model.train(
    data="railway-track-fault-detection",  # folder with subfolders per class
    epochs=50,
    imgsz=224,
    batch=64,
    task="classify"
)
```
> Ensure your dataset is structured for **classification** (e.g., one subfolder per class under `train/`, `val/` (optional), `test/` (optional)).

## Inference / Preview
```python
from ultralytics import YOLO
from PIL import Image

model = YOLO("runs/classify/train/weights/best.pt")
img = Image.open("path/to/sample.jpg")
pred = model(img)  # returns predictions
print(pred)
```

## Troubleshooting
- **OpenMP duplicate error on Windows**: Set environment var `KMP_DUPLICATE_LIB_OK=TRUE` as a quick workaround, or ensure only one OpenMP runtime is loaded.
- **Torch install issues**: Install the correct wheel for your CUDA/CPU from the official PyTorch site first, then install the rest.
- **Kaggle download**: If `kagglehub` fails, manually download the dataset and place it under `railway-track-fault-detection/`.

## License

This project is released under the [MIT License].

You are free to use, copy, modify, merge, publish, and distribute this work, provided that proper credit is given to the original author:

**© 2025 Léon Sobrie**

This repository is intended for **educational and research purposes** related to railway object detection and computer vision using YOLO.  
The software is provided *“as is”*, without warranty of any kind.

