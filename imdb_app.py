# Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
index = imdb.get_word_index()
ind_to_word = {value: key for key, value in index.items()}

# Load the pre-trained model with ReLU activation
imdb_model = load_model('simple_rnn_imdb.h5')

#Some Helper Functions:-
# Decoding of the review(Bits to Words):-
def decode(encoded_text):
    return ' '.join([ind_to_word.get(i - 3, '?') for i in encoded_text])

# Encoding of the review(Words to Bits):-
def preprocess(words):
    sent=words.lower().split()
    encoded_text = [index.get(word, 2) + 3 for word in sent]
    padded =sequence.pad_sequences([encoded_text], maxlen=500)
    return padded

# Creating a UI for Streamlit web-app:-
import streamlit as st
st.title("Movie Review Prediction Sytem")
st.write("Enter your comments and your reviews about a movie. Get your analysis with the help of confidence score:-")

input=st.text_area('Movie Review')
if st.button("Analyze.."):
    preprocessed_text=preprocess(input)
    pred=imdb_model.predict(preprocessed_text)
    if pred[0][0] > 0.65:
        sentiment = 'Positive Review'  
    elif pred[0][0] > 0.35 and pred[0][0] < 0.65:
        sentiment='Mixed Review'
    else:
        sentiment='Negative Review'

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence Score: {pred[0][0]}")

else:
    st.write("Please enter a comment/review")

