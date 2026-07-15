import cv2
import numpy as np
import joblib
import os
from insightface.app import FaceAnalysis

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("model/knn_arcface.pkl")

# ----------------------------
# Load Class Dictionary
# ----------------------------
class_dict = joblib.load("model/class_dictionary.pkl")

# Convert {name:number} -> {number:name}
class_number_to_name = {v: k for k, v in class_dict.items()}

# ----------------------------
# Load InsightFace App
# ----------------------------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 forces CPU


# ----------------------------
# Feature Extraction
# ----------------------------
def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    faces = face_app.get(img)

    if len(faces) == 0:
        return None

    # Use the largest detected face (in case of multiple faces)
    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )

    embedding = faces[0].embedding
    return embedding.reshape(1, -1)


# ----------------------------
# Prediction
# ----------------------------
def predict_celebrity(image_path):

    features = preprocess_image(image_path)

    if features is None:
        return None

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    sorted_probs = sorted(
        zip(class_number_to_name.values(), probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "prediction": class_number_to_name[prediction],
        "probabilities": sorted_probs
    }