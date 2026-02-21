import streamlit as st
# import os 
# from dotenv import load_dotenv
# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

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

#load llm (Ollama)
llm = Ollama(model="gemma2:2b")

# Chat Interface
user_question = st.text_input("Ask a question about C++: ")

if user_question:
    with st.spinner("Thinking..."):
        docs = db.similarity_search(user_question)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f""" 
        Answer the question using only the context below.

        context:
        {context}

        Quesstion:
        {user_question}

        Answer:
        """

        response = llm.invoke(prompt)

    st.subheader("Answer:")
    st.write(response)