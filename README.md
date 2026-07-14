# 🎭 Celebrity Image Classification using Machine Learning

A machine learning project that classifies celebrity faces from images using **Computer Vision**, **Wavelet Transform**, and **Support Vector Machine (SVM)**. The project also includes a **Streamlit web application** where users can upload an image and receive the predicted celebrity along with prediction probabilities and a sample image from the dataset.

---

# 📌 Features

- Upload any celebrity image
- Automatic face detection
- Eye detection for reliable face extraction
- Wavelet Transform feature extraction
- SVM-based classification
- Prediction probabilities
- Displays a similar image from the dataset
- Streamlit web interface

---

# 🛠️ Technologies Used

- Python
- OpenCV
- Haar Cascade Classifiers
- NumPy
- PyWavelets
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

---

# 📂 Project Structure

```
CelebrityClassification/
│
├── dataset/
│   ├── Lionel Messi/
│   ├── Maria Sharapova/
│   ├── Roger Federer/
│   ├── Serena Williams/
│   └── Virat Kholi/
│
├── test dataset/
│
├── model/
│   ├── celeb_classifier_model.pkl
│   ├── class_dictionary.pkl
│   ├── haarcascade_eye.xml
│   └── haarcascade_frontalface_default.xml
│
├── uploads/
│
├── util.py
├── main.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The project contains images of five celebrities.

| Celebrity | Images |
|-----------|-------:|
| Lionel Messi | 21 |
| Maria Sharapova | 22 |
| Roger Federer | 9 |
| Serena Williams | 18 |
| Virat Kohli | 11 |

Total Training Images: **81**

An additional unseen test dataset was used for evaluation.

---

# 🔄 Workflow

```
Image
   │
   ▼
Face Detection
   │
   ▼
Eye Detection
   │
   ▼
Crop Face
   │
   ▼
Resize (32×32)
   │
   ▼
Wavelet Transform
   │
   ▼
Feature Extraction
   │
   ▼
4096-Dimensional Feature Vector
   │
   ▼
SVM Classifier
   │
   ▼
Prediction
```

---

# 🖼️ Image Preprocessing

Each uploaded image undergoes the following preprocessing steps:

### 1. Face Detection

Haar Cascade detects all faces.

### 2. Eye Detection

Only faces with **at least two detected eyes** are accepted.

### 3. Face Cropping

The detected face is cropped.

### 4. Resize

The cropped face is resized to

```
32 × 32
```

### 5. Wavelet Transform

A Haar Wavelet Transform extracts texture information.

### 6. Feature Vector

The RGB image and wavelet image are stacked together.

Final feature vector size:

```
3072 + 1024 = 4096 Features
```

---

# 🤖 Model Training

Three machine learning models were evaluated using **GridSearchCV**.

## Models Compared

- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree

---

# Grid Search Results

| Model | Cross Validation Score |
|--------|----------------------:|
| Support Vector Machine | **0.7954** |
| Logistic Regression | 0.7886 |
| Decision Tree | 0.5547 |

Best model:

```
Support Vector Machine (Linear Kernel)
```

Best Parameters

```python
{
    'classifier__C': 1,
    'classifier__gamma': 'scale',
    'classifier__kernel': 'linear'
}
```

---

# 📈 Performance

### Training Accuracy

```
85.29%
```

### Test Accuracy (Unseen Images)

```
92.86%
```

---

# Confusion Matrix

```
[[9 0 0 0 0]
 [0 8 1 1 0]
 [0 0 5 0 0]
 [0 2 0 3 0]
 [1 0 0 0 3]]
```

---

# Prediction Example

```
Predicted Celebrity

Virat Kohli

Prediction Probabilities

Virat Kohli        60.00%
Roger Federer      22.30%
Lionel Messi       15.09%
Serena Williams     1.86%
Maria Sharapova     0.75%
```

---

# 🌐 Streamlit Web Application

The application allows users to

- Upload an image
- Detect the celebrity
- Display prediction probabilities
- Show a sample image from the dataset

Run the application:

```bash
streamlit run main.py
```

---

# ⚠️ Challenges Faced & Debugging

During development, several issues were encountered and resolved:

### 1. Face Detection

Some images contained multiple faces or no detectable faces.

**Solution**

Accepted only faces with **two detected eyes**.

---

### 2. Folder Name Mismatch

Example

```
Virat Kohli
```

vs

```
virat kholi
```

This caused

```
KeyError
```

**Solution**

Created a consistent class dictionary and mapping.

---

### 3. GridSearchCV Prediction

Attempting to use

```python
class_dict[prediction]
```

produced errors because `GridSearchCV` is not a dictionary.

**Solution**

Used

```python
best_estimator_
```

and maintained a reverse class mapping.

---

### 4. SVM Probability

Initially,

```python
predict_proba()
```

was unavailable.

**Solution**

Retrained the model using

```python
SVC(probability=True)
```

---

### 5. OpenCV Installation

Encountered

```
AttributeError:
module 'cv2' has no attribute 'CascadeClassifier'
```

Root cause

- Python 3.14 installed OpenCV 5.0 preview.

Solution

- Switched to Python 3.12
- Installed

```bash
opencv-python==4.10.0.84
```

---

### 6. Streamlit Environment

Virtual environment mismatch caused import issues.

Solution

Created a fresh virtual environment and reinstalled dependencies.

---

### 7. Probability Calibration Warning

Scikit-learn warning

```
probability=True
```

is deprecated.

Future improvement

Use

```python
CalibratedClassifierCV
```

instead.

---

# 🚀 Future Improvements

- Deep Learning (CNN)
- FaceNet embeddings
- MobileNetV3
- MTCNN face detector
- Real-time webcam prediction
- Multi-face detection
- Deploy on Streamlit Community Cloud
- Docker support
- REST API using FastAPI

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/Celebrity-Image-Classification.git
```

Move into the project

```bash
cd Celebrity-Image-Classification
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
streamlit run main.py
```

---

# 👨‍💻 Author

**soni kumari**

Machine Learning & Computer Vision Project

---

# ⭐ If you found this project useful, consider giving it a star!
