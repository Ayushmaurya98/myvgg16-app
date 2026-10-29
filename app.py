streamlit
tensorflow-cpu
pillow
numpy

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# 1. Page Configuration
st.set_page_config(
    page_title="VGG16 Image Classifier", 
    page_icon="🖼️", 
    layout="centered"
)

st.title("🖼️ VGG16 Image Classifier")
st.write("Upload any image (JPG, JPEG, PNG) to see what the VGG16 Deep Learning model detects.")

# 2. Cache the Model to prevent memory leaks and reloading overhead
@st.cache_resource
def load_vgg16_model():
    # Swap out for tf.keras.models.load_model('your_model.h5') if using a custom model file.
    return tf.keras.applications.VGG16('model_vgg16.h5')

with st.spinner("Loading VGG16 Architecture into memory..."):
    model = load_vgg16_model()

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
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Your Image")
        st.image(display_img, use_container_width=True)
        
    with col2:
        st.subheader("Top Predictions")
        with st.spinner("Running inference..."):
            # Execute Model Prediction
            predictions = model.predict(processed_img)
            
            # Decode the top 3 prediction probabilities
            decoded_results = decode_predictions(predictions, top=3)[0]
            
            # Render predictions dynamically with progress bars
            for index, (imagenet_id, label, prob) in enumerate(decoded_results):
                # Clean up ImageNet labels (e.g., 'golden_retriever' -> 'Golden Retriever')
                formatted_label = label.replace('_', ' ').title()
                percentage = prob * 100
                
                st.write(f"**{index + 1}. {formatted_label}**")
                st.progress(float(prob))
                st.caption(f"Confidence score: {percentage:.2f}%")
