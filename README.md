# Face Classification using MobileNet & Transfer Learning

A face classification system built with **MobileNet** (transfer learning), **face_recognition** for preprocessing, and a **Streamlit** web interface for real-time inference. The pipeline handles everything from raw image preprocessing and augmentation through model training to deployment via Docker.

## How It Works

The project has two main components:

**Training pipeline** (`face-recog-mobile.py`) takes a folder of labeled face images, resizes them to 224×224, validates that each image contains a detectable face (discarding those that don't), splits the data into train/validation/test sets (70/20/10), applies image augmentation (50 augmented copies per training image), and fine-tunes a MobileNet model with a custom classification head. The trained model is saved as `face-mobile.h5`.

**Inference app** (`app.py`) loads the trained model into a Streamlit web app where users can upload an image and get a predicted class label with confidence score.

## Project Structure

```
├── data/                    # Raw training images (one subfolder per person/class)
│   ├── person_1/
│   ├── person_2/
│   └── ...
├── processed/               # Auto-generated train/val/test splits + augmented data
├── no-face/                 # Images where no face was detected (auto-moved)
├── shells/                  # Shell scripts
├── face-recog-mobile.py     # Training pipeline
├── app.py                   # Streamlit inference app
├── face-mobile.h5           # Trained model weights (~16 MB)
├── accuracy_plot.png         # Training/validation accuracy over epochs
├── confusion_matrix.png     # Test set confusion matrix
├── Dockerfile               # Container config for deployment
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- `dlib` (required by `face_recognition` — may need CMake installed)

### Installation

```bash
git clone <repo-url>
cd face-recog-opencv

pip install -r requirements.txt
```

### Prepare Training Data

Organize your images under the `data/` directory with one subfolder per class:

```
data/
├── alice/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── bob/
│   ├── img1.jpg
│   └── ...
```

> **Note:** The final Dense layer in `face-recog-mobile.py` is set to 6 output units. Update `Dense(6, ...)` to match your actual number of classes before training.

## Usage

### Train the Model

```bash
python face-recog-mobile.py
```

This will:

1. Resize all images to 224×224
2. Detect and filter out images with no faces (moved to `no-face/`)
3. Split data into train, validation, and test sets
4. Generate 50 augmented images per training sample
5. Fine-tune MobileNet for 10 epochs
6. Save the model to `face-mobile.h5`
7. Output `accuracy_plot.png` and `confusion_matrix.png`

### Run the Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, upload a face image, and get the classification result.

### Run with Docker

```bash
docker build -t face-classifier .
docker run -p 8501:8501 face-classifier
```

## Model Architecture

The model uses **MobileNet** pretrained on ImageNet as a frozen feature extractor, with a custom head:

```
MobileNet (frozen) → GlobalAveragePooling2D → Dense(256, ReLU) → Dense(num_classes, Softmax)
```

Training uses Adam optimizer with categorical crossentropy loss. Image augmentation includes rotation, shifts, shear, zoom, and horizontal flips.

## Key Dependencies

| Package            | Purpose                              |
|--------------------|--------------------------------------|
| `tensorflow/keras` | Model training and inference         |
| `face_recognition` | Face detection and validation        |
| `streamlit`        | Web interface                        |
| `scikit-learn`     | Data splitting and evaluation        |
| `matplotlib`       | Accuracy plots                       |
| `seaborn`          | Confusion matrix visualization       |
| `Pillow`           | Image loading and resizing           |
| `opencv-python`    | Image processing utilities           |

## License

See repository for license details.

