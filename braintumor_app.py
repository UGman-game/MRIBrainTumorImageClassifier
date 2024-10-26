import streamlit as st
import base64
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance

st.set_page_config(page_title="Brain Tumor Classification", page_icon=":brain:", layout="wide")

model = tf.keras.models.load_model('/content/drive/MyDrive/Models/brain_tumor.h5')

# Define the class labels
class_labels = ['pituitary', 'invalid', 'meningioma', 'glioma']

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to apply blur to the background image
def apply_blur(image):
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    return blurred_image

# Streamlit app
def main():
    st.markdown(
        """
        <style>
        .stApp {
            background: url('https://i.postimg.cc/QNPYnn1D/braintwo.jpg');
            background-size: cover;
        }
        .stApp > div:first-child {
            filter: blur(15px);
        }
        .result-box {
            border: 2px solid white;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        .result-box h2 {
            margin-top: 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Brain Tumor Classification")
    st.subheader("A simple app to classify brain tumors")

    # Text input for patient's name
    patient_name = st.text_input("Enter patient's name")

    # File uploader for image input
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Add a submit button
    submit_button = st.button("Submit")

    # Add a clear button
    clear_button = st.button("Clear")

    if uploaded_file is not None:
        # Display the uploaded image
        image = load_img(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if submit_button:
            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Add a progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()

            # Make the prediction
            prediction = model.predict(preprocessed_image)
            predicted_index = np.argmax(prediction)
            predicted_label = class_labels[predicted_index]

            # Update the progress bar
            progress_bar.progress(100)
            progress_text.text("Done!")

            # Display the prediction
            st.subheader("Result:")
            with st.container():
                with st.container() as result_box:
                    st.markdown(
                        f"<h2>{predicted_label}</h2>",
                        unsafe_allow_html=True
                    )
                    for label, score in zip(class_labels, prediction[0]):
                        st.write(f"- {label}: {score:.1f}")

        if clear_button:
            uploaded_file = None

        # Display the patient's name
        st.subheader("Patient's Name:")
        st.write(patient_name)

# Run the app
if __name__ == "__main__":
    main()
