import streamlit as st
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(cleaned_tokens)

# Load the QA pipeline model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=-1)

# Function to answer a question using the QA model
def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Function to split the report into sections
def split_report_into_sections(text):
    sections = re.split(r'\n\s*\n', text)  # Split based on double newline
    section_dict = {}
    for section in sections:
        if '\n' in section:
            heading, content = section.split('\n', 1)
            section_dict[heading.lower()] = content
    return section_dict

# Function to get the relevant section based on the question
def get_relevant_section(question, section_dict):
    for heading in section_dict:
        if any(word in heading for word in question.lower().split()):
            return section_dict[heading]
    return " ".join(section_dict.values())  # If no match, search the whole report

# Main function with sections
def main_with_sections(pdf_file, question):
    # Step 1: Extract and split the report into sections
    text = extract_text_from_pdf(pdf_file)
    section_dict = split_report_into_sections(text)

    # Step 2: Get the relevant section based on the question
    relevant_section = get_relevant_section(question, section_dict)

    # Step 3: Answer the question using the QA model
    answer = answer_question(question, relevant_section)

    return answer

# Streamlit Interface
st.title("PDF Question Answering System")

st.write("Upload a PDF file and ask a question. The system will analyze the text and provide an answer.")

# Upload PDF
uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

# Ask Question
question = st.text_input("Ask a Question", placeholder="e.g., What is the company's profit after tax?")

# Process PDF and Question
if uploaded_pdf is not None and question:
    answer = main_with_sections(uploaded_pdf, question)
    st.text_area("Answer", value=answer, height=200)
