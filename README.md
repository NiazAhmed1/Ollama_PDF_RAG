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

2. Run the Streamlit app
   ```bash
   streamlit run app.py
 3. Upload a PDF: Use the file uploader to upload your PDF document. The pages of the PDF will be displayed as images.
 4. Choose a Model: Select one of the locally available models from Ollama.
 5. Ask Questions: Interact with the PDF content by typing your queries in the chat box. The system will retrieve relevant information and provide concise answers.


## Example Usage
1. **Upload a PDF document**: Start by browsing a PDF document you’d like to explore.
2. **Select a Model**: From the dropdown menu, select a model available locally in Ollama that you want to use to query the PDF content.
3. **Enter a Query**: In the chat box, type a question, such as:
   - **Example 1**: "What is mentioned in section 3?"
   - **Example 2**: "Summarize the main findings of the document."
   - **Example 3**: "List any key terms introduced in the first chapter."

   The system will respond with relevant information extracted directly from the PDF content.

## Project Structure
- `app.py`: The main Streamlit application file that handles file uploading, model selection, querying, and displaying results in the chat format.
- `requirements.txt`: Contains the list of Python libraries needed to run the project, including Langchain, Streamlit, and other dependencies.

This project is designed to make document interaction seamless and user-friendly. Whether for research, document review, or educational purposes, Ollama PDF RAG offers an intuitive way to engage with PDF content.

