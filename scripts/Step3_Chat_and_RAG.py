# RAG Medical Chatbot Setup - Step 3 (Groq Version)
# =================================================
# Uses Groq-hosted Mixtral/LLaMA3 for answering medical queries from anatomy PDFs.

from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# -----------------------------
# 🔐 Load API keys from .env
# -----------------------------
load_dotenv()
groq_api_key = os.getenv("API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

# -----------------------------
# 📂 Load FAISS vector store
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)

faiss_path = "../vectorstore/faiss_index"
vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -----------------------------
# 🧠 Groq LLM Setup (Mixtral or LLaMA3)
# -----------------------------
llm = ChatGroq(
    temperature=0.2,
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"  # or "llama3-70b-8192"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -----------------------------
# 🧪 Ask medical questions
# -----------------------------
while True:
    query = input("\n🩺 Ask a medical/anatomy question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"query": query})
    print("\n💬 Answer:")
    print(result["result"])

