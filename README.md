# Dress Compliance Violation Detection using YOLOv8

A Flask-based web application for detecting dress compliance violations using YOLOv8. The system analyzes images, videos, and live webcam streams to identify whether a person complies with predefined clothing regulations. Detection results are displayed with annotated bounding boxes.

## Features

- Image detection
- Video detection
- Live webcam detection
- Automatic PDF report generation on Video Detection
- YOLOv8 object detection
- Flask web interface

## Dataset

This project was trained using the **clothes classfy** object detection dataset from Roboflow Universe.

- **Dataset:** clothes classfy
- **Author:** zhaozhao
- **Platform:** Roboflow Universe
- **Task:** Object Detection
- **Classes:** Tshirt, Dress, Jacket, Pants, Shirt, Short, Skirt, Sweater
- **License:** CC BY 4.0

The dataset is not included in this repository. Please download it from the original Roboflow Universe project.

## Tech Stack

- Python
- Flask
- Ultralytics YOLOv8
- OpenCV
- NumPy
- ReportLab

```

## Project Structure
.
├── app.py
├── requirements.txt
├── YOLOV8_Deteksi_AturanBerpakaianMahasiswa_GeraldoTan.ipynb
├── templates/
├── venv/
├── static/
├── output/
├── best.pt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/geraldotan/YOLOV8-for-Dress-Compliance-Detection.git
cd <YOLOV8-for-Dress-Compliance-Detection>
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

## Demo

<img width="1440" height="900" alt="Tangkapan Layar 2026-06-30 pukul 22 56 17" src="https://github.com/user-attachments/assets/f9aed834-1ab9-4ab2-b4f5-018233e3fb11" />
<img width="1440" height="900" alt="Tangkapan Layar 2026-06-30 pukul 22 21 26" src="https://github.com/user-attachments/assets/fdedb74f-48c1-463c-a749-972349a73b2e" />

## Reproducibility

This repository contains a reorganized and updated implementation of the published research project.

The original development notebooks used during the research were not preserved. The included notebook demonstrates the current training and inference workflow for reproducibility purposes.

Due to differences in software versions, dependencies, hardware, and training environments, reproduced evaluation metrics may differ from those reported in the publication.

## Publication

This repository accompanies a published research project. The implementation has been organized for public release and educational purposes.

## License

This project is licensed under the MIT License.
