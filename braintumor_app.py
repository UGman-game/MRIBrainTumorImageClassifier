import streamlit as st
import base64
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
 
model = load_model('brain_tumor.h5')
 

def preprocess_image(image):
     image = Image.open(image)
     image = image.convert('RGB')
     image = ImageEnhance.Brightness(image).enhance(1.2)
     image = ImageEnhance.Contrast(image).enhance(1.2)
     image = image.resize((128, 128))
     image = np.array(image) / 255.0
     image = np.expand_dims(image, axis=0)
     return image
 
labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
 

def app():
# Set page background image
    folder_name = "images"
    file_name = "bg.jpg"
    image_path = os.path.join(folder_name, file_name)
    page_bg_img = '''
       <style>
       body {
       background-image: url("data:image/png;base64,%s");
       background-size: cover;
       }
       </style>
    ''' % base64.b64encode(open(image_path, 'rb').read()).decode()
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    st.title('Brain Tumor Classification')
    st.write('This app can classify the type of brain tumor from an uploaded image.')
    
    name = st.text_input('Enter Patient name')
    if name:
        st.write('Patient Name :, ' + name )
    
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = preprocess_image(uploaded_file)
        prediction = model.predict(image)
        predicted_label = labels[np.argmax(prediction)]
        st.write('Type of tumor:', predicted_label)
        st.image(image[0], caption='Uploaded Image', use_column_width=True)
 
if __name__ == '__main__':
     app()
