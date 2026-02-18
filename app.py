# app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os

st.set_page_config(page_title="Resume RAG Chatbot", layout="wide")
st.title("Ask My Resume - LLMs & HuggingFace with Chat Memory")

# Upload resume
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file:
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Loading and splitting your resume...")

    # Load PDF
    loader = PyPDFLoader("temp_resume.pdf")
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)

    st.info(f"Document split into {len(texts)} chunks.")

    # Initialize embeddings
    st.info("Initializing embeddings (local)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # retrieve top 3 chunks

    # Initialize local LLM
    st.info("Loading local LLM (Flan-T5)... this may take ~30s on first run...")
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Memory to store chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Conversational Retrieval Chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False  # set True if you want to show sources
    )

    # Initialize session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Input box
    query = st.text_input("Ask something about your resume:")

    if query:
        with st.spinner("Generating answer..."):
            result = qa({"question": query})
            answer = result["answer"]

            # Store in session
            st.session_state.messages.append({"question": query, "answer": answer})

    # Display chat history
    if st.session_state.messages:
        st.subheader("Chat History")
        for i, chat in enumerate(st.session_state.messages):
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Bot:** {chat['answer']}")
            st.markdown("---")
