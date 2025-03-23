# Animal Detection System

A Python-based application that detects and classifies animals in images and videos using YOLOv8 and a custom classification model. The system specifically highlights carnivorous animals with red bounding boxes for safety awareness.

## Features

- Real-time animal detection in images and videos
- Classification of various animal species
- Special highlighting of carnivorous animals with red bounding boxes
- Adjustable confidence threshold
- User-friendly GUI interface

## Requirements

- Python 3.8+
- OpenCV
- TensorFlow
- Ultralytics (YOLOv8)
- PIL (Python Imaging Library)
- Tkinter (usually comes with Python)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Animal_Detection.git
cd Animal_Detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the required models:
- Place your custom model file (`my_model.keras`) in the project directory
- The YOLOv8 model will be downloaded automatically on first run

## Usage

1. Run the application:
```bash
python animal_detection_app.py
```

2. Use the GUI to:
   - Open images or videos
   - Adjust confidence threshold
   - View detections with color-coded boxes (red for carnivorous animals)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Download Links

### Model Files
- Trained Model: [Download](https://drive.google.com/file/d/1HQjc1Stn5B69-WPTPBy3qZrY1a9aSN9w/view?usp=drive_link)

### Project Files
- Jupyter Notebook: [Download](https://colab.research.google.com/drive/1zDfpLmk7SPK-ss0tm0Z1UlRORxe9ZzrL?usp=sharing)
- Dataset: [Download](https://www.kaggle.com/datasets/anthonytherrien/image-classification-64-classes-animal)

Note: Please ensure you have sufficient storage space before downloading. The model files are large in size. 