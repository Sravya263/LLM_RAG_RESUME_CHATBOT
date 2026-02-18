
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Resume RAG Chatbot")
st.title("🧠 Ask My Resume - LLM + RAG")

# Check API Key
if not os.getenv("OPENAI_API_KEY"):
    st.error("Please set your OPENAI_API_KEY in the .env file")
    st.stop()

# Upload resume
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file:
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp_resume.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    query = st.text_input("Ask something about your resume:")

    if query:
        response = qa.run(query)
        st.subheader("Answer:")
        st.write(response)
