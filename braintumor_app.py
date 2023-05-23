import streamlit as st
import base64
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
 
model = tf.keras.models.load_model('brain_tumor.h5')

# Define the class labels
class_labels = ['pituitary', 'notumor', 'meningioma', 'glioma']

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
def main():
    st.title("Brain Tumor Classification")

    # Text input for patient's name
    patient_name = st.text_input("Enter patient's name")

    # File uploader for image input
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = load_img(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make the prediction
        prediction = model.predict(preprocessed_image)
        predicted_index = np.argmax(prediction)
        predicted_label = class_labels[predicted_index]

        # Display the prediction
        st.subheader("Prediction:")
        if predicted_label == 'notumor':
            st.write("No tumor detected")
        else:
            st.success(f"Tumor Detected: {predicted_label}")

        # Display the patient's name
        st.subheader("Patient's Name:")
        st.write(patient_name)

# Run the app
if __name__ == "__main__":
    main()
