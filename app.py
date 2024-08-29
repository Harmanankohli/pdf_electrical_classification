import re
import io
import nltk
import time
import string
import random
import pickle
import threading
import numpy as np
import pandas as pd
import urllib.request
import streamlit as st
nltk.download('punkt')
from tika import parser
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download("punkt_tab")
nltk.download('stopwords')
from functools import partial
from nltk.corpus import stopwords
from xgboost import XGBClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textacy.preprocessing import replace
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

# class for timeout error
class TimeoutError(Exception):
    pass

# function for monitoring the execution time of getData function
def monitor_execution_time(func, args=(), kwargs={}, timeout=120):
    result = []
    error = []
    
    def target():
        try:
            result.append(func(*args, **kwargs))
        except Exception as e:
            error.append(e)
    
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        thread.join(1)  # Give the thread a second to clean up
        raise TimeoutError(f"Error: Function execution exceeded {timeout} seconds")
    
    if error:
        raise error[0]
    
    return result[0]

# Function for reading the pdf
def getData(URL):
    try:
        # Set up headers to mimic a browser request
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        # Fetch the PDF content from the URL
        req = urllib.request.Request(URL, headers=headers)
        remote_file = urllib.request.urlopen(req).read()
        
        # Create a BytesIO object from the remote file
        remote_file_bytes = io.BytesIO(remote_file)
        
        # Parse the PDF content with Tika
        parsed_pdf = parser.from_buffer(remote_file_bytes)
        data = parsed_pdf['content']
        
        return data
    except Exception as e:
        return f"Error: {str(e)}"
    
# function for getting the pdf
def process_pdf_with_timeout(url, timeout=120):
    try:
        return monitor_execution_time(getData, args=(url,), timeout=timeout)
    except TimeoutError:
        return f"Error: Processing exceeded {timeout} seconds"
    except Exception as e:
        return f"Error: {str(e)}"

# function for preprocessing the text
def preprocessing(text):
    '''Function for performing the preprocessing of the text'''
     # Convert text to lowercase to ensure uniformity
    text = text.lower()
    # remobing urls
    text = replace.urls(text,'')
    # Remove tabs, digits, newlines, and specific punctuation marks like quotes and dashes
    text = re.sub(r"[\t\d\n'“”„-]+", "", text)
    # Remove all non-alphabetic characters except for spaces
    text = re.sub(r"[^a-z\s]",'',text)
    # Remove any remaining punctuation using str.translate and string.punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Strip leading and trailing whitespace from the text
    text = text.strip()
    # Initialize the set of stopwords in English
    stop_words = set(stopwords.words('english'))
     # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Tokenize the text into words
    words = word_tokenize(text)
    # Filter out stopwords and apply lemmatization
    text = ' '.join([lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words])
    return text

# function to convert the characters to numeric format
def vectorize(text):
    """function to vectorizing the text into its numeric representation"""
    # Load the vectorizer
    with open('vectorizer.pkl','rb') as file:
        vectorizer=pickle.load(file) 
    vect_text = vectorizer.transform([text]).toarray()
    return vect_text

def getPrediction(vect_text):
    '''Function for making the prediction'''
    #Load the Model
    with open('model.h5','rb') as file:
        model=pickle.load(file)
    result = model.predict(vect_text)
    # Load the labelencoder
    with open('labelencoder.pkl','rb') as file:
        le=pickle.load(file) 
    result = le.inverse_transform(result)
    return result


if __name__ == '__main__':
    spinner_messages = [
        "Maybe it's worth a million unicorns? Just kidding about the unicorns. (Unless...?) 🤸",
        "Hang tight! We're working faster than a squirrel with a nut stash full of caffeine. 🤸",
        "Coffee break? Nah, gotta get this done for you, champ! 🤸",
    ]
    # streamlit app
    st.title("Automated Classification of Electrical Product PDFs")
    st.write("**Note: This app classifies the given pdf into electrical categories - fuses, lighting, cables, others. Please provide a pdf belonging to one of these categories**")
    url=st.text_input("Enter the url for pdf","")
    st.write("**Search any pdf related to the above listed categories and paste the same in the above input field and then see the magic.**")
    timeout = st.text_input("Enter the timeout for the pdf processing in seconds", 120)
    if st.button("Predict Class"):
        with st.spinner(text=random.choice(spinner_messages)):
            text = process_pdf_with_timeout(url,int(timeout))
            if text.startswith("Error:"):
                st.error(f'Please try for another pdf url. Following error is encountered: {str(text)}', icon="🚨")
            else:
                processed_text = preprocessing(text)
                vect_text = vectorize(processed_text)
            result = getPrediction(vect_text)
            st.write("**The given PDF is classified into: "+str(*result).upper()+"**")
            st.write("Following are the contents of the PDF:")
            st.write(text) 
        
