import streamlit as st
from PIL import Image
import tempfile
import os
import random

from util import predict_celebrity

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Celebrity Image Classification",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 Celebrity Image Classification")
st.write("Upload an image to predict the celebrity.")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, width='stretch')

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        image_path = tmp_file.name

    # Predict
    result = predict_celebrity(image_path)

    # Delete temporary file
    os.remove(image_path)

    with col2:

        if result is None:
            st.error("❌ No face with two eyes detected.")
        else:

            prediction = result["prediction"]
            probabilities = result["probabilities"]

            st.success(f"### 🏆 Predicted Celebrity: {prediction}")

            st.subheader("Prediction Probabilities")

            for name, prob in probabilities:
                st.write(f"**{name}**")
                st.progress(float(prob))
                st.write(f"{prob*100:.2f}%")

            # -----------------------------
            # Display Similar Dataset Image
            # -----------------------------

            st.subheader("Similar Image From Dataset")

            dataset_path = os.path.join("dataset", prediction)

            if os.path.exists(dataset_path):

                images = [
                    img for img in os.listdir(dataset_path)
                    if img.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                if images:

                    sample_image = random.choice(images)

                    st.image(
                        os.path.join(dataset_path, sample_image),
                        caption=prediction,
                        width='stretch'
                    )
                else:
                    st.warning("No images found in dataset folder.")
            else:
                st.warning("Dataset folder not found.")