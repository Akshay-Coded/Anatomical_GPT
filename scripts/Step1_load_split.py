# RAG Medical Chatbot Setup - Step 1
# ==================================
# This script loads 4 anatomy-related PDF books, extracts the text, splits it into manageable chunks, and prepares the data for embedding.


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import json
import os

# -----------------------------
# 📁 Step 1: Load Medical PDFs
# -----------------------------
# Adjust paths to where your actual PDFs are stored
pdf_dir = os.path.join(os.path.dirname(__file__), '../Dataset')
pdf_paths = [
    os.path.join(pdf_dir, "clinically_oriented_anatomy.pdf"),
    os.path.join(pdf_dir, "grays_anatomy.pdf"),
    os.path.join(pdf_dir, "Sobotta_anatomy_1.pdf"),
    os.path.join(pdf_dir, "Sobotta_anatomy_2.pdf")
]

all_documents = []

for path in tqdm(pdf_paths, desc="📄 Loading PDFs"):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        continue

    loader = PyPDFLoader(path)  # Uses PyMuPDF internally
    docs = loader.load()
    all_documents.extend(docs)  # Append Document objects from each PDF

print(f"✅ Loaded {len(all_documents)} total pages from all PDFs.")

# -----------------------------
# ✂️ Step 2: Chunk Text
# -----------------------------
# Use RecursiveCharacterTextSplitter to chunk content into ~500 token segments
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(tqdm(all_documents, desc="✂️ Chunking text"))
print(f"✅ Split into {len(chunks)} chunks. Ready for embedding.")

# Save chunks for inspection or testing
with open("../embeddings/chunks_preview.txt", "w", encoding="utf-8") as f:
     for i, chunk in enumerate(tqdm(chunks[:10], desc="💾 Saving preview chunks")):
         f.write(f"--- Chunk {i+1} ---\n{chunk.page_content}\n\n")

# Save all chunks to disk as JSON
os.makedirs("../embeddings", exist_ok=True)

chunk_texts = [chunk.page_content for chunk in chunks]

with open("../embeddings/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunk_texts, f, ensure_ascii=False, indent=2)

print("✅ All chunks saved to ../embeddings/chunks.json")