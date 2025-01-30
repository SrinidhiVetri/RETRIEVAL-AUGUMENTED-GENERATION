import streamlit as st
import faiss
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# OpenAI API Key
OPENAI_API_KEY = "your-api-key"
openai.api_key = OPENAI_API_KEY  # Set OpenAI API Key

# Load Sentence Transformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle None values
    return text.strip()

# Function to Chunk Text
def chunk_text(text, chunk_size=500):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to Create FAISS Index
def create_faiss_index(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True).astype(np.float32)  # Convert to float32
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

# Function to Retrieve Relevant Chunks
def retrieve_top_k(query, index, chunks, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

# Function to Generate Answers using OpenAI GPT-4
def generate_answer(query, context):
    prompt = f"Based on the following research content, answer concisely:\n\nContext:\n{context}\n\nQuery: {query}\n\nAnswer:"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI research assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response["choices"][0]["message"]["content"]

# Streamlit UI
st.title("📄 AI Research Paper Q&A (RAG System)")

uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        if not pdf_text:
            st.error("No text extracted from the PDF. Try another document.")
        else:
            text_chunks = chunk_text(pdf_text)
            index, chunks = create_faiss_index(text_chunks)
            st.success("PDF processed successfully! You can now ask questions.")

    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Retrieving relevant information..."):
            relevant_texts = retrieve_top_k(query, index, chunks)
            context = "\n".join(relevant_texts)
            answer = generate_answer(query, context)
            st.write("### 🤖 Answer:")
            st.write(answer)

            st.write("### 📚 Retrieved Context:")
            for i, text in enumerate(relevant_texts, 1):
                st.write(f"**Chunk {i}:** {text}")
