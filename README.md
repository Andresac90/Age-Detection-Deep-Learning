# Age Detection with Deep Learning

Age detection system using OpenCV and pre-trained deep learning models. Detects faces from webcam and predicts age ranges.

Based on the GeeksforGeeks tutorial: https://www.geeksforgeeks.org/age-detection-using-deep-learning-in-opencv/

---

## What It Does

- Detects faces in real-time from webcam
- Predicts age ranges for detected faces
- Works with 8 age categories: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)

---

## Setup

### Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Download Models

Run the download script to get all required models:

```bash
python download_models.py
```

This will download and extract:
- Face detection models (opencv_face_detector)
- Age prediction model (age_net.caffemodel)

Total download size: ~50 MB

### Verify Setup

Test that everything is working:

```bash
python test_system.py
```

This checks OpenCV installation and webcam access.

---

## Usage

```bash
python age_detection.py --mode webcam
```

Controls:
- `q` - quit
- `s` - save current frame

---

## Project Structure

```
Age-Detection-Deep-Learning/
├── age_detection.py           # Main application
├── download_models.py         # Model downloader
├── test_system.py             # System verification
├── requirements.txt           # Python dependencies
├── .gitignore
├── models/                    # Downloaded models
│   ├── opencv_face_detector.pbtxt
│   ├── opencv_face_detector_uint8.pb
│   ├── age_deploy.prototxt
│   └── age_net.caffemodel
└── README.md
```

---

## How It Works

**Face Detection**
- Uses OpenCV DNN with SSD architecture
- Input: 300x300 image
- Detects faces with 70% confidence threshold

**Age Prediction**
- Modified VGG-16 CNN trained on Adience dataset
- Input: 227x227 face crop
- Outputs probability for 8 age ranges

---

## Command-Line Options

```
--mode          webcam, image, or video (default: webcam)
--input         Input file path (for image/video modes)
--output        Output file path (optional)
--camera        Camera index (default: 0)
--face-proto    Face detection config file
--face-model    Face detection weights
--age-proto     Age prediction config file
--age-model     Age prediction weights
```

---

## Troubleshooting

**OpenCV GUI Error**

If you see `The function is not implemented` error:

```bash
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python==4.10.0.84
```

**Camera Not Found**

Try different camera index: `--camera 1` or `--camera 2`

**Models Missing**

Re-run the download script: `python download_models.py`

**No Faces Detected**

- Check lighting
- Face camera directly
- Move closer to camera

---

## Technical Details

**Models Used:**
- Face Detection: OpenCV DNN (TensorFlow)
- Age Classification: Caffe model (VGG-16 based)

**Training Data:**
- Adience dataset with ~26,000 face images

**Research:**
Levi, G., & Hassner, T. (2015). Age and Gender Classification using Convolutional Neural Networks. CVPR Workshops.

---

## Limitations

- Predicts age ranges, not exact ages
- Accuracy varies with lighting and angle
- Works best on frontal faces
- May struggle with poor lighting or occlusions

---

## Author

Andrés Acevedo  
Toronto, Canada

GitHub: https://github.com/Andresac90  
LinkedIn: https://www.linkedin.com/in/andresacep/

---

## License

MIT License

---

## Acknowledgments

- GeeksforGeeks tutorial
- OpenCV community
- Gil Levi and Tal Hassner for the age/gender models
- Adience dataset creators