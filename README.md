# Face Recognition MobileNet Project

This project trains a face recognition image classification model using **MobileNet** as a transfer learning base model. It processes face image datasets, removes images where no face is detected, creates train/validation/test splits, applies image augmentation, trains a Keras model, evaluates performance, and saves visual result plots.

## Project Overview

The main script is:

```text
face-recog-mobile.py
```

The script performs the following tasks:

1. Reads image data from the `data/` directory.
2. Resizes all images to `224 x 224` pixels.
3. Detects faces using the `face_recognition` library.
4. Moves images with no detected face into the `no-face/` directory.
5. Splits valid face images into training, validation, and testing sets.
6. Creates an augmented training dataset.
7. Uses a pre-trained **MobileNet** model for feature extraction.
8. Adds custom classification layers.
9. Trains the model.
10. Saves the trained model as `face-mobile.h5`.
11. Generates accuracy and confusion matrix plots.

## Technologies Used

- Python
- Keras / TensorFlow
- MobileNet
- face_recognition
- Pillow
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn

## Folder Structure

Before running the project, your dataset should be arranged like this:

```text
project-folder/
│
├── face-recog-mobile.py
├── data/
│   ├── person_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   │
│   ├── person_2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   │
│   └── person_3/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
```

After running the script, the following folders/files will be created automatically:

```text
project-folder/
│
├── no-face/
│   └── images where no face was detected
│
├── processed/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── train_aug/
│
├── face-mobile.h5
├── accuracy_plot.png
└── confusion_matrix.png
```

## Dataset Format

Each person/class should have a separate folder inside the `data/` directory.

Example:

```text
data/
├── Rahat/
├── John/
├── Alex/
├── Maria/
├── David/
└── Sara/
```

Each folder should contain multiple face images of that person.

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

For Windows PowerShell:

```bash
venv\Scripts\Activate.ps1
```

For Windows Command Prompt:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pillow matplotlib seaborn scikit-learn tensorflow keras face-recognition
```

> Note: The `face_recognition` package depends on `dlib`, which may require additional system setup, especially on Windows.

## How to Run

Place your dataset inside the `data/` folder, then run:

```bash
python face-recog-mobile.py
```

## Model Architecture

The project uses **MobileNet** with ImageNet weights as the base model.

The original top layer is removed and replaced with custom layers:

```text
MobileNet base model
GlobalAveragePooling2D
Dense layer with 256 units and ReLU activation
Dense output layer with Softmax activation
```

The output layer in the current script is configured for **6 classes**:

```python
Dense(6, activation='softmax')
```

If your dataset has a different number of people/classes, update this value to match your number of classes.

Example:

```python
Dense(number_of_classes, activation='softmax')
```

## Training Configuration

The dataset is split into:

| Dataset | Ratio |
|---|---:|
| Training | 70% |
| Validation | 20% |
| Testing | 10% |

The model is trained for:

```text
10 epochs
```

The training images are augmented using:

- Rotation
- Width shift
- Height shift
- Shear
- Zoom
- Horizontal flip

The script generates **50 augmented images per original training image**.

## Output Files

After successful execution, the project produces:

| File | Description |
|---|---|
| `face-mobile.h5` | Trained Keras model file |
| `accuracy_plot.png` | Training and validation accuracy graph |
| `confusion_matrix.png` | Confusion matrix for test results |
| `no-face/` | Images where no face was detected |
| `processed/` | Processed train, validation, test, and augmented datasets |

## Important Notes

### 1. Update the number of classes

The script currently uses:

```python
Dense(6, activation='softmax')
```

Change `6` according to the number of folders/classes inside your `data/` directory.

### 2. Deprecated Keras methods

The script uses:

```python
model.fit_generator()
model.predict_generator()
```

In newer TensorFlow/Keras versions, these can be replaced with:

```python
model.fit()
model.predict()
```

### 3. Face detection requirement

Only images where a face is detected are used for training. Images without detected faces are moved to the `no-face/` folder.

### 4. Dataset quality matters

For better accuracy, use:

- Clear face images
- Good lighting
- Multiple angles
- Enough images per person
- Balanced number of images for each class

## Possible Improvements

- Automatically detect the number of classes instead of manually setting `Dense(6)`.
- Save class labels to a separate file for prediction use.
- Add a separate prediction script for new images.
- Replace deprecated Keras methods with current methods.
- Improve train/validation/test splitting so image filenames stay correctly matched with labels.
- Add command-line arguments for dataset path, epochs, batch size, and output model name.

## Example Prediction Workflow

After training, the saved model can be loaded and used to classify new face images. A future prediction script may follow this flow:

1. Load `face-mobile.h5`.
2. Load and resize a new image to `224 x 224`.
3. Normalize the image by dividing pixel values by `255`.
4. Pass the image into the model.
5. Get the predicted class with the highest probability.
6. Map the predicted class index back to the person name.

## License

This project is for educational and research purposes. You may modify and improve it according to your requirements.
