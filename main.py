

from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
import faiss

# --- Configuration ---
openai.api_key = "key here"  # Replace this with your actual API key
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Helper Functions ---
def extract_pdf_text(file_path):
    return extract_text(file_path)

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

def generate_quiz(context):
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

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# --- Main Logic ---
if __name__ == "__main__":
    print("📄 PDF-to-Quiz Generator")
    pdf_path = input("Enter the path to your PDF file: ").strip()

    text = extract_pdf_text(pdf_path)
    print("✅ PDF text extracted.")

    chunks = split_into_chunks(text)
    print(f"✅ Split into {len(chunks)} chunks.")

    embeddings = embed_chunks(chunks, embedding_model)
    index = create_faiss_index(np.array(embeddings))

    query = input("🔍 Enter a topic or keyword to generate quiz from: ").strip()
    relevant_chunks = search_similar_chunks(query, index, chunks, embedding_model)
    context = "\n".join(relevant_chunks)

    print("🧠 Generating quiz...")
    quiz = generate_quiz(context)
    print("\n🎯 Generated Quiz:\n")
    print(quiz)
