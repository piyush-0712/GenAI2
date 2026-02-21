import streamlit as st
# import os 
# from dotenv import load_dotenv
# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# page config
st.set_page_config(page_title="C++ RAG Chatbot")
st.title("C++ RAG Chatbot")
st.write("Ask any question related to C++ Introduction")

# cache document loading
@st.cache_resource
def load_vectorstore():
    # Load document
    loader = TextLoader("C++_Introduction.txt", encoding="utf-8")
    documents = loader.load()
    
    # Split text
    text_Splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, 
        chunk_overlap=20
        )
    final_documents = text_Splitter.split_documents(documents)
    
    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
        )
    
    # Create FAISS Vector Store
    db = FAISS.from_documents(final_documents, embeddings)
    return db

# Load vector DB(only once)
db = load_vectorstore()

# User Input 
query = st.text_input("Enter your question for C++ : ")

if query:
    docs = db.similarity_search(query, k=3)
    st.subheader("RetrievedContext:")
    for i, doc in enumerate(docs):
        st.markdown(f"**Result {i+1}:**")
        st.write(doc.page_content)