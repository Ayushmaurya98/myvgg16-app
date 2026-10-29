import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

# 1. Page Configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier", 
    page_icon="🐱🐶", 
    layout="centered"
)

st.title("🐱🐶 Cat vs Dog VGG16 Classifier")
st.write("Upload an image to see if the model detects a Cat or a Dog.")

# 2. Cache the Custom Model
@st.cache_resource
def load_vgg16_model():
    # Make sure 'model.h5' is pasted in the same folder as this app.py file
    return tf.keras.models.load_model('model_vgg16.h5')

with st.spinner("Loading your custom VGG16 model into memory..."):
    try:
        model = load_vgg16_model()
    except Exception as e:
        st.error(f"Could not load 'model.h5'. Make sure the file is uploaded to GitHub in the same folder. Error: {e}")
        st.stop()

# 3. Image Preprocessing Pipeline
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Store copy for user interface display
    display_img = img.copy()
    
    # VGG16 strictly requires 224x224 input images
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)
    
    return display_img, processed_img

# 4. Streamlit UI Elements
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process the uploaded image
    display_img, processed_img = preprocess_image(uploaded_file)
    
    # Create two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Image")
        st.image(display_img, use_container_width=True)
        
    with col2:
        st.subheader("Prediction Result")
        with st.spinner("Running inference..."):
            # Execute Model Prediction
            predictions = model.predict(processed_img)
            
            # --- CUSTOM CAT AND DOG MAPPING ---
            # Define your classes exactly in the order your model outputs them.
            # Usually index 0 is Cat and index 1 is Dog, or vice versa.
            CLASS_NAMES = ['Cat', 'Dog'] 
            
            # If your model outputs a single probability value (Binary/Sigmoid)
            if predictions.shape[-1] == 1:
                prob = predictions[0][0]
                if prob > 0.5:
                    result_label = "Dog"
                    confidence = prob * 100
                else:
                    result_label = "Cat"
                    confidence = (1 - prob) * 100
            # If your model outputs two probability values (Softmax)
            else:
                predicted_class_idx = np.argmax(predictions[0])
                result_label = CLASS_NAMES[predicted_class_idx]
                confidence = predictions[0][predicted_class_idx] * 100
            
            # Display output nicely
            st.success(f"Detected: **{result_label}**")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
