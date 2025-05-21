import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
import faiss

# --- Configuration ---
openai.api_key = "your-openai-api-key"  # Replace with your actual API key
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Helper Functions ---
def extract_pdf_text(file):
    return extract_text(file)

def split_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def embed_chunks(chunks, model):
    return model.encode(chunks)

def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_similar_chunks(query, index, chunks, model, top_k=5):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)
    return [chunks[i] for i in indices[0]]

def generate_quiz(context, use_mock=False):
    if use_mock:
        return f"""
Q1. What is the main idea of the provided content?
A. Option A
B. Option B
C. Option C
D. Option D
Correct Answer: B

Q2. Which concept was discussed in the passage?
A. Alpha
B. Beta
C. Gamma
D. Delta
Correct Answer: C

Q3. What can be inferred from the study material?
A. Statement A
B. Statement B
C. Statement C
D. Statement D
Correct Answer: D

Q4. Which example best illustrates the point made?
A. Sample 1
B. Sample 2
C. Sample 3
D. Sample 4
Correct Answer: A

Q5. What is the conclusion?
A. Point 1
B. Point 2
C. Point 3
D. Point 4
Correct Answer: B
"""

    prompt = f"""
    You are an expert teacher. Based on the following study material, create 5 multiple-choice quiz questions with 4 options each, and mark the correct answer.

    STUDY MATERIAL:
    {context}

    FORMAT:
    Q1. ...
    A. ...
    B. ...
    C. ...
    D. ...
    Correct Answer: ...
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except openai.error.OpenAIError as e:
        return f"⚠️ Error generating quiz: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Quiz Generator", layout="centered")
st.title("📘 PDF to Quiz Chatbot")

use_mock = st.sidebar.checkbox("🔌 Use Mock Generator (Offline Mode)", value=False)

uploaded_pdf = st.file_uploader("Upload your study material (PDF)", type=["pdf"])

if uploaded_pdf:
    with st.spinner("Extracting and embedding content..."):
        text = extract_pdf_text(uploaded_pdf)
        chunks = split_into_chunks(text)
        embeddings = embed_chunks(chunks, embedding_model)
        index = create_faiss_index(np.array(embeddings))
    st.success("✅ PDF processed!")

    user_query = st.text_input("🔍 Enter a topic or concept for quiz generation:")
    if user_query:
        with st.spinner("Searching content and generating quiz..."):
            relevant = search_similar_chunks(user_query, index, chunks, embedding_model)
            context = "\n".join(relevant)
            quiz = generate_quiz(context, use_mock=use_mock)
        st.markdown("### 📝 Generated Quiz")
        st.code(quiz)
