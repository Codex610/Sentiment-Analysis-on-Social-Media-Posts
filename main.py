import os 
import pickle
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

app = Flask(__name__)

# Load model + tokenizer
model = load_model("lstm_model.h5")
with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]

    return " ".join(words)

def predict_sentiment(text):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)

    prediction = model.predict(padded)[0][0]

    sentiment = "Positive" if prediction > 0.5 else "Negative"

    return sentiment, float(prediction)


# Route
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        text = request.form.get("text")
        if text:
            result, confidence = predict_sentiment(text)

    return render_template("index.html", result=result, confidence=confidence)


if __name__=="__main__":
    app.run(debug=True)
