# 🧠 Resume RAG Chatbot

A simple **LLM + Retrieval-Augmented Generation (RAG)** application that lets you **chat with your resume**.  
Upload a PDF resume, ask questions, and get accurate, grounded answers using vector search and LLMs.

Built with:
- LangChain
- OpenAI
- FAISS
- Streamlit

---

## ✨ Features

- 📄 Upload your resume (PDF)
- ✂️ Automatically splits the document into chunks
- 🧠 Generates embeddings for each chunk
- 🗂️ Stores embeddings in FAISS vector database
- 🔍 Retrieves relevant chunks for each question
- 💬 LLM generates context-aware, grounded answers
- ⚡ Simple UI with Streamlit

---

## 🚀 How It Works

1. Upload your resume (PDF)
2. The document is split into smaller chunks
3. Each chunk is converted into embeddings
4. Embeddings are stored in FAISS
5. User question retrieves the most relevant chunks
6. The LLM uses those chunks to generate the final answer

---

## 🛠️ Tech Stack

- **LangChain** – RAG pipeline orchestration  
- **OpenAI** – Embeddings + LLM  
- **FAISS** – Vector database for similarity search  
- **Streamlit** – Web UI  

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
-git clone https://github.com/your-username/resume-rag-chatbot.git
cd resume-rag-chatbot
pip install -r requirements.txt
Run the app using command
streamlit run app.py
