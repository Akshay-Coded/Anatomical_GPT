# RAG Medical Chatbot Setup - Step 2
# ==================================
# This script generates embeddings for the text chunks and stores them in a FAISS vector database.

# -----------------------------
# 🔧 Prerequisites
# -----------------------------
# pip install faiss-cpu sentence-transformers langchain

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import json

# -----------------------------
# 📥 Load Chunked Data
# -----------------------------
# load the JSON chunks and storing locally for further operations

with open("../embeddings/chunks.json", "r", encoding="utf-8") as f:
    chunk_texts = json.load(f)

from langchain.schema import Document
documents = [Document(page_content=text) for text in chunk_texts]
# -----------------------------
# 🧠 Step 1: Load Embedding Model
# -----------------------------
model_name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

embedding_model = HuggingFaceEmbeddings(
    model_name=model_name
)

# -----------------------------
# 📦 Step 2: Store in FAISS
# -----------------------------
print("🔄 Creating FAISS vector store...")
vectorstore = FAISS.from_documents(documents, embedding_model)

# Save the vector store
faiss_path = "../vectorstore/faiss_index"
os.makedirs(faiss_path, exist_ok=True)
vectorstore.save_local(faiss_path)

print(f"✅ FAISS index saved at {faiss_path}")
