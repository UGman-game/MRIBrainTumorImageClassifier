 import streamlit as st
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
     st.title('Brain Tumor Classification')
     st.write('This app can classify the type of brain tumor from an uploaded image.')
     uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
 
     if uploaded_file is not None:
         image = preprocess_image(uploaded_file)
         prediction = model.predict(image)
         predicted_label = labels[np.argmax(prediction)]
         st.write('Type of tumor:', predicted_label)
         st.image(image[0], caption='Uploaded Image', use_column_width=True)
 

 if __name__ == '__main__':
     app()
