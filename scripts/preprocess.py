import os
import re
import pickle
import pandas as pd
import nltk
from PIL import Image
import pytesseract
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Define a lemmatizer
lemmatizer = WordNetLemmatizer()

# Extend the stopwords list
stopwords = list(nltk_stopwords.words('english'))

# Define path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update this path as per your tesseract installation

def preprocess_text(text):
    """
    Preprocesses the input text by converting to lowercase, removing URLs, HTML tags,
    special characters, retweet tags, and stopwords, and lemmatizing the words.
    
    Args:
    text (str): The text to preprocess.
    
    Returns:
    str: The preprocessed text.
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    tokens = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stopwords])
    return text

def preprocess_image(image_path):
    """
    Preprocesses the input image by converting to grayscale, resizing,
    and extracting text using OCR.
    
    Args:
    image_path (str): The path to the image to preprocess.
    
    Returns:
    str: The preprocessed text extracted from the image.
    """
    try:
        img = Image.open(image_path)
        img = img.convert('L')
        img = img.resize((256, 256))
        extracted_text = pytesseract.image_to_string(img)
        preprocessed_text = preprocess_text(extracted_text)
        return preprocessed_text
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""

def preprocess_csv_data(input_file, output_file):
    """
    Preprocesses the CSV data by loading the dataset, removing rows with missing
    values or empty text, and preprocessing the text.
    
    Args:
    input_file (str): The path to the input CSV file.
    output_file (str): The path to the output CSV file where preprocessed data will be saved.
    
    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    df = pd.read_csv(input_file, usecols=[2, 3], header=None, names=['sentiment', 'text'])
    df.dropna(inplace=True)
    df = df[df['text'].apply(len) > 1]
    df['text'] = df['text'].apply(preprocess_text)
    df.to_csv(output_file, index=False)
    return df

def save_vectorizer(vectorizer, filename):
    """
    Saves the vectorizer to a file.
    
    Args:
    vectorizer (TfidfVectorizer): The vectorizer to save.
    filename (str): The path to the file where the vectorizer will be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(vectorizer, f)

def save_data(data, filename):
    """
    Saves the data to a file.
    
    Args:
    data: The data to save.
    filename (str): The path to the file where the data will be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def main():
    """
    Main function to preprocess the CSV data, perform train-test split,
    vectorize the text data, and save the preprocessed data and vectorizer.
    """
    try:
        # Define file paths
        input_file = os.path.join('..', 'data', 'twitter_training.csv')
        output_file = os.path.join('..', 'data', 'preprocessed_twitter_training.csv')
        
        # Preprocess CSV data
        df = preprocess_csv_data(input_file, output_file)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)
        
        # Vectorize text data
        vectorizer = TfidfVectorizer(stop_words=stopwords)
        X_train_vect = vectorizer.fit_transform(X_train)
        X_test_vect = vectorizer.transform(X_test)
        
        # Save the vectorizer and data
        vectorizer_filename = os.path.join('..', 'models', 'tfidf_vectorizer.pkl')
        save_vectorizer(vectorizer, vectorizer_filename)
        
        train_data_filename = os.path.join('..', 'data', 'train_data.pkl')
        test_data_filename = os.path.join('..', 'data', 'test_data.pkl')
        save_data((X_train_vect, y_train), train_data_filename)
        save_data((X_test_vect, y_test), test_data_filename)
        
        print("Preprocessing completed and data saved.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
