import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('my_model_6_classes.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0 
    return img_array

def main():
    st.title("Image Classification App")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        test_image = preprocess_image(uploaded_file)
        
        predictions = model.predict(test_image)
        
        predicted_class = np.argmax(predictions, axis=1)
        
        class_names = ["class_0", "class_1", "class_2", "class_3", "class_4", "class_5"]
        
        st.success(f"Predicted Class: {class_names[predicted_class[0]]}")
    
    st.sidebar.title("About")
    st.sidebar.info(
        "This is a simple image classification app using TensorFlow and Streamlit.\n"
        "Upload an image, and the app will predict its class."
    )

if __name__ == "__main__":
    main()
