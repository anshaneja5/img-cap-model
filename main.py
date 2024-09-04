import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import DenseNet201
import numpy as np
import pickle
from PIL import Image, ImageOps

# Load the saved model
model = tf.keras.models.load_model('model (2).keras')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Initialize the DenseNet201 model for feature extraction
densenet = DenseNet201(include_top=False, weights='imagenet', pooling='avg')

def extract_features(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = tf.keras.applications.densenet.preprocess_input(image)
    feature = densenet.predict(image, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length=74):
    feature = extract_features(image)
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        
        y_pred = model.predict([feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        
        word = idx_to_word(y_pred, tokenizer)
        
        if word is None:
            break
        
        in_text += " " + word
        
        if word == 'endseq':
            break
    
    return in_text.replace("startseq", "").replace("endseq", "").strip()

st.title("Image Captioning App")

image_source = st.radio("Choose image source:", ("Upload Image", "Take Photo"))

if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)  
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Generate Caption'):
            caption = predict_caption(model, image, tokenizer)
            st.write(f"Generated Caption: {caption}")

elif image_source == "Take Photo":
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture)
        image = ImageOps.exif_transpose(image) 
        st.image(image, caption='Captured Image', use_column_width=True)
        
        if st.button('Generate Caption'):
            caption = predict_caption(model, image, tokenizer)
            st.write(f"Generated Caption: {caption}")
