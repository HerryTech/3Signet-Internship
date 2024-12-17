import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

# Cache the model to avoid reloading it on every run
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('optimized_model.h5')
    return model

# Load the model with a spinner
with st.spinner('Model is being loaded...'):
    model = load_model()

# Title of the app
st.write("""
         # Image Classification Model
         """)

# Upload image file
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

# Preprocessing and prediction function
def import_and_predict(image_data, model):
    size = (32, 32)  # Resize image to match model's input shape
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = np.asarray(image).astype('float32') / 255.0  # Normalize to [0, 1]
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    img_reshape = img[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(img_reshape)  # Predict
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    # Display uploaded image
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict and display results
    predictions = import_and_predict(image, model)
    class_names = [
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
        "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup",
        "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
        "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
        "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid",
        "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
        "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
        "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
        "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train",
        "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
    ]
    
    # Get predicted class and confidence
    predicted_class = class_names[np.argmax(predictions)]
    confidence_score = np.max(predictions)

    # Display results
    string = f"This image most likely is: **{predicted_class}** with a confidence score of **{confidence_score:.2f}**"
    st.success(string)
