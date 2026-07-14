import cv2
import numpy as np
import pywt
import joblib
import os

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("model/celeb_classifier_model.pkl")

# ----------------------------
# Load Class Dictionary
# ----------------------------
class_dict = joblib.load("model/class_dictionary.pkl")

# Convert {name:number} -> {number:name}
class_number_to_name = {v: k for k, v in class_dict.items()}

# ----------------------------
# Load Haar Cascades
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    os.path.join("model", "haarcascade_frontalface_default.xml")
)

eye_cascade = cv2.CascadeClassifier(
    os.path.join("model", "haarcascade_eye.xml")
)

# Verify cascades loaded correctly
if face_cascade.empty():
    raise FileNotFoundError(
        "Could not load haarcascade_frontalface_default.xml"
    )

if eye_cascade.empty():
    raise FileNotFoundError(
        "Could not load haarcascade_eye.xml"
    )

# ----------------------------
# Wavelet Transform
# ----------------------------
def w2d(img, mode='haar', level=1):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    gray /= 255

    coeffs = pywt.wavedec2(gray, mode, level=level)

    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    img_har = pywt.waverec2(coeffs_H, mode)
    img_har *= 255
    img_har = np.uint8(img_har)

    return img_har


# ----------------------------
# Crop Face if Two Eyes Found
# ----------------------------
def get_cropped_image_if_2_eyes(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:
            return roi_color

    return None


# ----------------------------
# Feature Extraction
# ----------------------------
def preprocess_image(image_path):

    cropped_img = get_cropped_image_if_2_eyes(image_path)

    if cropped_img is None:
        return None

    scaled_raw = cv2.resize(cropped_img, (32, 32))

    img_har = w2d(cropped_img, 'db1', 5)
    scaled_har = cv2.resize(img_har, (32, 32))

    combined_img = np.vstack((
        scaled_raw.reshape(32 * 32 * 3, 1),
        scaled_har.reshape(32 * 32, 1)
    )).reshape(1, 4096)

    return combined_img


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