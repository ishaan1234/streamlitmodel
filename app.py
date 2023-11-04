import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(r"plant.h5") 

st.title('Plant')
st.write('Plant checker')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    image = Image.open(uploaded_file)
    image = image.resize((224, 224))  
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch

    # Make prediction
    prediction = model.predict(image)
    pred = np.argmax(prediction, axis=1)
    st.write(pred)
    

