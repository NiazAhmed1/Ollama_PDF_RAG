import os
import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from typing import List, Dict, Any, Tuple
import logging
import ollama
import pdfplumber

from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
import chromadb.api
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


chromadb.api.client.SharedSystemClient.clear_system_cache()


st.set_page_config(
    page_title="Ollama PDF RAG",
    page_icon='bollon',
    layout='wide',
    initial_sidebar_state='collapsed'
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
# Initialize chat history as an empty list
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



def extract_model_names(model_info:Dict[str,List[Dict[str,Any]]]) -> Tuple[str,...]:
    logger.info("Extracting model names from model_info")
    model_names=tuple(model["name"] for model in model_info["models"])
    logger.info(f"Extracting model names : {model_names}")
    return model_names


# Function to handle file upload
def upload_pdf_file():
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file is not None:
        temp_dir = "temp"
        file_path = os.path.join(temp_dir, uploaded_file.name)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    return None


# Function to convert PDF pages to images
def convert_pdf_to_images(pdf_path):
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            image = page.to_image()
            images.append(image.original)
    return images



# Retriever function
def retriever(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorStore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorStore.as_retriever()
    
    return retriever


# Final response function
def final_response(retriever, llm, query):
    
    model=Ollama(model=llm)
    template = """You are an assistant for question-answering tasks.  
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer: """
    
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()} 
        | prompt 
        | model
        | StrOutputParser() 
    )
    result = rag_chain.invoke(query)
    return result


# Streamlit app layout
def main():
    st.subheader("🧠 Ollama PDF RAG", divider="gray", anchor=False)

    model_info = ollama.list()
    available_models = extract_model_names(model_info)

    # Create two columns
    col1, col2 = st.columns([1.5, 2])
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    
    # In col1, handle PDF upload and retriever creation
    with col1:
        uploaded_file_path = upload_pdf_file()
        
        if uploaded_file_path:
            ret=retriever(uploaded_file_path)
            zoom_level = st.slider("Zoom Level", min_value=100, max_value=1000, value=700, step=50)
        
            with st.container(height=410, border=True):
                st.subheader("PDF Pages as Images")
                images = convert_pdf_to_images(uploaded_file_path)
                for img in images:
                    st.image(img, width=zoom_level)
        if available_models:
            selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", available_models)
                    
    with col2:
        message_container = st.container(height=500, border=True)
        prompt = st.chat_input("Enter a prompt here...")
        if prompt:
            if uploaded_file_path is None:
                st.warning("Please upload a PDF file before entering a prompt.")
            else:
                response = final_response(ret, selected_model, prompt)
                st.session_state.chat_history.append((prompt, response))

        # Display the entire conversation history
        with message_container:
            for i, (user_prompt, bot_response) in enumerate(st.session_state.chat_history):
                st.write(f"😎 {user_prompt}")
                st.write(f"🤖 {bot_response}")
                st.divider()


        

if __name__ == "__main__":
    main()