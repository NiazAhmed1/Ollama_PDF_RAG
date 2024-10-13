# Ollama_PDF_RAG

 Ollama PDF RAG with Langchain and Streamlit
This project implements PDF Retrieval-Augmented Generation (RAG) using Langchain and Streamlit. It allows users to upload PDF files, select a locally available model through Ollama, and interact with the content of the PDF by querying it in a chat interface. The responses are generated based on the content of the uploaded PDF.

The user interface (UI) is built using Streamlit, making it easy and interactive for users to upload PDFs, select models, and chat with the document.

🛠️ Features
PDF Upload: Upload any PDF document, which is displayed below the uploader as a series of images (one image per page).
Model Selection: Choose from available models in Ollama installed on your local system to process the queries.
Conversational Interface: Ask questions related to the PDF content, and receive concise, context-aware responses. The interaction is displayed in a chat-like format.
🚀 Getting Started
Prerequisites
Ensure you have the following installed on your system:

Python 3.7+
pip (Python package manager)
Installation
Clone the repository:

git clone https://github.com/NiazAhmed1/Ollama_PDF_RAG.git
cd Ollama_PDF_RAG

Install dependencies from the requirements.txt file:
pip install -r requirements.txt

Set up Ollama: Ensure you have the necessary models available locally in Ollama. This project assumes you have Ollama installed and configured.

Running the Application

Navigate to the project directory in your terminal:
cd path/to/your/project

Run the Streamlit app:
streamlit run app.py

Upload a PDF: Use the file uploader to upload your PDF document. The pages of the PDF will be displayed as images.

Choose a Model: Select one of the locally available models from Ollama.

Ask Questions: Interact with the PDF content by typing your queries in the chat box. The system will retrieve relevant information and provide concise answers.

Example Usage
Upload a PDF document.
Select an Ollama model from the dropdown menu.
Enter a query, such as "What is mentioned in section 3?", and the system will respond with the relevant information from the PDF.
📦 Project Structure
app.py: The main Streamlit application file that handles file uploading, model selection, querying, and displaying results.
requirements.txt: Contains the list of Python libraries needed to run the project.
temp/: Temporary directory where uploaded PDF files are stored.
🤝 Contributing
Feel free to open issues or submit pull requests. Contributions are always welcome!
