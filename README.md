# 🎭 Celebrity Image Classification

A Streamlit web application that predicts the celebrity in an uploaded image using **InsightFace (ArcFace)** for feature extraction and a **K-Nearest Neighbors (KNN)** classifier trained on facial embeddings.

Supported celebrities:

- Lionel Messi
- Maria Sharapova
- Roger Federer
- Serena Williams
- Virat Kohli

---

# 🚀 Live Demo

🔗 **Live App:** https://celebrity-image-classification.streamlit.app/

🔗 **GitHub Repository:** https://github.com/sonibteceai/Celebrity-Image-Classification

---

# 📂 Project Structure

```text
Celebrity-Image-Classification/
│
├── main.py                         # Streamlit application
├── util.py                         # Prediction & preprocessing functions
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── knn_arcface.pkl             # Trained KNN model
│   ├── class_dictionary.pkl        # Class dictionary
│   └── buffalo_l/                  # InsightFace pretrained model
│
├── dataset/
│   ├── Lionel Messi/
│   ├── Maria Sharapova/
│   ├── Roger Federer/
│   ├── Serena Williams/
│   └── Virat Kohli/
│
├── uploads/                        # Temporary uploaded images
│
└── pretrain_model_folder/
    ├── pretrained_model.ipynb      # Training notebook
    ├── evaluation.ipynb            # Model evaluation notebook
    ├── knn_arcface.pkl
    └── class_dictionary.pkl
```

---

# 🧠 Machine Learning Pipeline

```text
Input Image
      │
      ▼
InsightFace (buffalo_l)
      │
      ▼
SCRFD Face Detection
      │
      ▼
Face Alignment
      │
      ▼
ArcFace
(512-D Face Embedding)
      │
      ▼
K-Nearest Neighbors (KNN)
      │
      ▼
Celebrity Prediction
      │
      ▼
Prediction Probabilities
```

---

# 🔬 Model Training

Instead of manually designing image features, the project uses **ArcFace**, a pretrained face recognition model.

Each detected face is converted into a **512-dimensional embedding**, which is then used to train multiple machine learning classifiers.

The following models were compared using **GridSearchCV (5-Fold Cross Validation)**:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest
- Decision Tree
- Gaussian Naive Bayes

The best-performing model was selected for deployment.

---

# 📊 Model Comparison

| Model | Cross Validation Score | Test Accuracy |
|--------|-----------------------:|--------------:|
| **KNN** | **96.8%** | **97.8%** |
| Logistic Regression | 96.8% | 95.7% |
| SVM | 96.5% | 95.7% |
| Random Forest | 96.5% | 96.8% |
| Gaussian Naive Bayes | 95.7% | 96.8% |
| Decision Tree | 84.4% | 87.1% |

---

# 🔄 Project Evolution

## Version 1 (Classical Computer Vision)

### Face Detection

- Haar Cascade
- Eye Detection
- Required at least two visible eyes

### Feature Extraction

- Raw Image (32×32)
- Wavelet Transform
- Combined into a 4096-dimensional feature vector

### Classifier

- Support Vector Machine (SVM)

### Limitations

- Sensitive to lighting conditions
- Failed when eyes were partially closed
- Failed on profile faces
- Lower accuracy on unseen images

---

## Version 2 (Deep Learning + Machine Learning)

### Face Detection

- InsightFace (SCRFD Detector)

### Face Alignment

- Automatic facial landmark alignment

### Feature Extraction

- ArcFace Pretrained Model
- 512-dimensional embeddings

### Classifier

- K-Nearest Neighbors (KNN)

### Advantages

- Better accuracy
- Robust to pose variations
- Works under different lighting conditions
- No eye detection required
- Faster inference
- Smaller feature vector (512 vs 4096)

---

# 🌐 Streamlit Web Application

The web application allows users to:

- Upload an image
- Detect the face automatically
- Predict the celebrity
- Display prediction probabilities
- Show a related celebrity image from the dataset

---

# 🛠 Technologies Used

- Python
- Streamlit
- InsightFace
- ArcFace
- OpenCV
- NumPy
- Scikit-learn
- Joblib
- Pillow

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/sonibteceai/Celebrity-Image-Classification.git

cd Celebrity-Image-Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run main.py
```

---

# 📈 Retraining the Model

To train the model on new data:

1. Add images to the `dataset` folder.
2. Open `pretrain_model_folder/pretrained_model.ipynb`.
3. Generate ArcFace embeddings.
4. Run GridSearchCV.
5. Save the best model.

```python
joblib.dump(best_model, "knn_arcface.pkl")
joblib.dump(class_dict, "class_dictionary.pkl")
```

Copy these files into the `model/` directory before running the Streamlit app.

---

# 🐞 Challenges & Debugging

During development, several issues were encountered and resolved:

- Fixed OpenCV installation issues (`CascadeClassifier` errors).
- Migrated from Haar Cascade to InsightFace for better detection.
- Switched from handcrafted wavelet features to ArcFace embeddings.
- Compared six machine learning classifiers using GridSearchCV.
- Solved model serialization using Joblib.
- Fixed Streamlit deployment issues.
- Used `opencv-python-headless` for cloud deployment compatibility.
- Improved prediction confidence using classifier probabilities.

---

# 📌 Future Improvements

- Add more celebrities.
- Support multiple face detection in one image.
- Fine-tune a deep learning classifier.
- Add Grad-CAM or embedding visualization.
- Display Top-3 predictions with confidence.
- Use FAISS for fast nearest-neighbor search on larger datasets.
- Add webcam support.

---
# 🚀 Deployment Notes

The application is deployed on **Streamlit Community Cloud**.

### Deployment Configuration

- Uses `opencv-python-headless` in `requirements.txt` to avoid installing unnecessary GUI components directly.
- However, **InsightFace** has a dependency on the standard `opencv-python`, which pip installs automatically as a transitive dependency.
- A `packages.txt` file is included to install the required Linux system libraries that OpenCV depends on during runtime.

### packages.txt

```text
libgl1
libglib2.0-0t64
```

### Why these packages are required

#### `libgl1`

Fixes the error:

```text
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

#### `libglib2.0-0t64`

Fixes the error:

```text
ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory
```

> **Note:** The correct package name is **`libglib2.0-0t64`**, not `libglib2.0-0`.

Recent Debian releases (used by Streamlit Community Cloud) renamed this package during the 64-bit `time_t` transition. Using the older package name can result in unresolved dependency errors involving `libffi7` or `libpcre3`.

### First Launch

The first deployment (cold start) may take **30–60 seconds** because InsightFace automatically downloads the pretrained **`buffalo_l`** model weights. These are cached and reused on subsequent runs.
# 👨‍💻 Author

**Soni B.Tech ECE AI Student**

GitHub: https://github.com/sonibteceai

---

# ⭐ If you like this project

Please consider giving the repository a **Star ⭐**.
