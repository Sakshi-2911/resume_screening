import streamlit as st
import pickle
import re

# Load models & vectorizer
with open("svc_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

st.title("Resume Screening App")
st.write("Upload a resume (plain text) to classify it")

uploaded = st.file_uploader("Choose a file", type=["txt"])
if uploaded is not None:
    text = uploaded.read().decode("utf-8")
    # Basic cleaning
    text_clean = re.sub(r'[^a-zA-Z ]', " ", text.lower())
    vec = tfidf.transform([text_clean])
    pred = model.predict(vec)[0]
    st.success(f"Prediction: {pred}")
