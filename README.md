
# 🧠 Resume RAG Chatbot

A simple LLM + Retrieval-Augmented Generation (RAG) project built using:

- LangChain
- OpenAI
- FAISS
- Streamlit

## 🚀 How It Works

1. Upload your resume (PDF)
2. The document is split into chunks
3. Chunks are converted into embeddings
4. Stored in FAISS vector database
5. User question retrieves relevant chunks
6. LLM generates grounded answer

## 🔧 Setup

### 1. Install dependencies

pip install -r requirements.txt

### 2. Create .env file

Create a file named `.env` and add:

OPENAI_API_KEY=your_openai_api_key_here

### 3. Run the app

streamlit run app.py

---

## 📌 Example Questions

- What technologies do I know?
- What cloud platforms have I used?
- What projects are mentioned?
- What is my education background?

---

Built for learning LLM + RAG fundamentals 🚀
# chatbot-using-LLMs-Hugging-face
