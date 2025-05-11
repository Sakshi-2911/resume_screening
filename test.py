# Enhanced Resume Categorizer Streamlit App

import os
import pandas as pd
import pickle
import re
import streamlit as st
from pypdf import PdfReader
from docx import Document
from io import BytesIO

# Load models
word_vector = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Title and description
st.set_page_config(page_title="Resume Categorizer", layout="centered")
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📄 AI Resume Categorizer")
st.write("Upload multiple resumes in **PDF or DOCX** format. The app will categorize them using a pre-trained ML model.")

# Utility Functions

def clean_resume(txt):
    clean_text = re.sub('http\S+\s', ' ', txt)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+\s', ' ', clean_text)
    clean_text = re.sub('@\S+', ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7F]+', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text.strip()

category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
    1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and Fitness",
    19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate",
}

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text.strip()

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def categorize_resumes(uploaded_files):
    results = []
    zip_buffer = BytesIO()

    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split(".")[-1].lower()
        text = ""

        if ext == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif ext == "docx":
            text = extract_text_from_docx(uploaded_file)

        if text:
            cleaned = clean_resume(text)
            features = word_vector.transform([cleaned])
            prediction = model.predict(features)[0]
            category = category_mapping.get(prediction, "Unknown")

            results.append({"Filename": uploaded_file.name, "Category": category})

    return results

# Upload & Output Interface
uploaded_files = st.file_uploader("📤 Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("🔍 Categorize Resumes"):
    if uploaded_files:
        results = categorize_resumes(uploaded_files)
        if results:
            df = pd.DataFrame(results)
            st.success("Resumes categorized successfully!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Categorization CSV",
                data=csv,
                file_name="categorized_resumes.csv",
                mime="text/csv",
            )
        else:
            st.warning("No text found in uploaded documents.")
    else:
        st.error("Please upload at least one file.")
