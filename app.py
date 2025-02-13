import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# variables
maxLength = 500

# word index
wordIndex = imdb.get_word_index()
reverseWordIndex = {value: key for key, value in wordIndex.items()}

# Load the updated model
model_path = os.path.abspath("simpleRNN-IMDB-v2.h5")
model = load_model("simpleRNN-IMDB-v2.h5")

# Function To Decode Reviews
def decodeReviews(encodedReview): 
    return ' '.join([reverseWordIndex.get(i - 3,'?') for i in encodedReview])

# Function to preprocess user-input
def preprocessText(text):
    words = text.lower().split()
    encodedReview = [wordIndex.get(word, 2) + 3 for word in words]
    paddedReview = sequence.pad_sequences([encodedReview], maxlen = maxLength)
    return paddedReview

# Prediction Function
def predictSentiment(review):
    preprocessedInput = preprocessText(review)
    prediction = model.predict(preprocessedInput)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit Application
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a Movie Review to classify as Positive or Negative")

# Ask for User Input
userInput = st.text_area("Movie Review")

# Predict The Results
if st.button('Classify'):
    preprocessedInput   = preprocessText(userInput)
    prediction          = model.predict(preprocessedInput)
    sentiment           = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display The Results
    st.write(f'Sentiment:           {sentiment}')
    st.write(f'Prediction Score     {prediction}')
else:
    st.write('Please Enter a Movie Review')
