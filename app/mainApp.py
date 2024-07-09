import streamlit as st
import pickle
import pytesseract
from PIL import Image
import re

# Set the tesseract executable path (update this to the correct path on your system)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load the trained model
model = pickle.load(open('../models/best_model.pkl', 'rb'))

# Load the vectorizer
vectorizer = pickle.load(open('../models/tfidf_vectorizer.pkl', 'rb'))

# Function to preprocess text
def preprocess_text(text):
    if not text:
        return None  # Return None if text is empty
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and punctuation
    text = re.sub(r'^RT[\s]+', '', text)  # Remove retweet tags
    return text.strip()  # Remove leading/trailing whitespace

st.title('CYBERBULLYING/HATE SPEECH PREDICTION')

# Text input field for entering tweet
tweet_input = st.text_input('Enter your tweet')

# Image uploader
image = st.file_uploader('Upload an image', type=['jpg', 'png'])

# Button to trigger predictions
submit = st.button('Predict')

if submit:
    if tweet_input:
        # Preprocess and vectorize the text input
        preprocessed_text = preprocess_text(tweet_input)
        vectorized_text = vectorizer.transform([preprocessed_text])
        
        # Predictions based on text input
        prediction = model.predict(vectorized_text)
        st.write('Prediction for text input:', prediction[0])

    if image:
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Use OCR to extract text from the uploaded image
        extracted_text = pytesseract.image_to_string(Image.open(image))
        
        # Display the extracted text
        st.write('Extracted text from the image:', extracted_text)

        # Preprocess and vectorize the extracted text
        preprocessed_text = preprocess_text(extracted_text)
        
        if preprocessed_text:
            vectorized_text = vectorizer.transform([preprocessed_text])
            
            # Predictions based on extracted text from the image
            image_prediction = model.predict(vectorized_text)
            st.write('Prediction for image text:', image_prediction[0])
        else:
            st.write('No text extracted from the image. Please try again.')
