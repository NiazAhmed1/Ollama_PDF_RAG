# Ollama PDF RAG with Langchain and Streamlit

This project implements PDF Retrieval-Augmented Generation (RAG) using Langchain and Streamlit. It allows users to upload PDF files, select a locally available model through Ollama, and interact with the content of the PDF in a chat interface. Responses are generated based on the content of the uploaded PDF, providing an interactive experience for users.

## Features
- **PDF Upload**: Upload any PDF document, which is displayed below the uploader as a series of images (one image per page).
- **Model Selection**: Choose from locally available models in Ollama to process queries.
- **Conversational Interface**: Ask questions related to the PDF content and receive context-aware responses in a chat-like format.

## Getting Started

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.7+
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/NiazAhmed1/Ollama_PDF_RAG.git
   cd Ollama_PDF_RAG
2. Install dependencies from the requirements.txt file
   ```bash
   pip install -r requirements.txt
 3. Set up Ollama: Make sure you have the necessary models available locally. This project assumes you have Ollama installed and configured.

### Running the Application
1. Navigate to the project directory in your terminal
```bash
   cd path/to/your/project/Ollama_PDF_RAG
