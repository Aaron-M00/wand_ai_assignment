# Intelligent Document Processing API

A modular FastAPI-based backend for **AI-powered document processing**, **text extraction**, and **intelligent data handling** using state-of-the-art models like **LangChain**, **Transformers**, and **PyTorch**.

This project is designed to process and understand unstructured documents, preparing them for advanced AI-driven analysis. It includes a well-organized architecture with separate modules for the server, utilities, app logic, and model integration.

---

## Features

-  **Document Ingestion** — Handles multiple file types (PDF, DOCX, images, etc.)  
-  **LangChain Integration** — Powers LLM-based document understanding  
-  **Model Support** — Supports transformer-based and vision-based models  
-  **FastAPI Backend** — Clean and production-ready RESTful API  
-  **Vector Store Support (ChromaDB)** — Enables semantic search and embeddings  
-  **Text Parsing** — Uses PyPDF, Unstructured, and Python-docx  
-  **Modular Design** — Separation of concerns across files for scalability  
-  **OpenAI + Local Model Ready** — Works with local or API-based models  

---

## Project Structure

📂 project-root/
1. app.py # Main application entry point
2. main.py # Script to launch or test the app
3. server.py # FastAPI server configuration
4. utils.py # Helper functions for file handling and processing
5. requirements.txt # Python dependencies
6. README.md # Project documentation


---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

2. Create and activate a virtual environment

python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

3. Install dependencies

pip install -r requirements.txt

### Running the API Server

Start the FastAPI server using Uvicorn:

uvicorn server:app --reload

The API will be available at:

http://127.0.0.1:8000
To view the interactive documentation, visit:

http://127.0.0.1:8000/docs

### Design Decisions

FastAPI was chosen for its async capabilities and automatic OpenAPI documentation.

LangChain and ChromaDB were used to handle embeddings and conversational retrieval efficiently.

Unstructured and PyMuPDF were selected for document parsing due to their strong performance on varied input formats.

The architecture separates concerns:

app.py: Core logic

server.py: API definition

utils.py: Helper functions

main.py: Entry or orchestration script

Extensibility First — Each module can be expanded independently (e.g., adding OCR, new parsers, or AI endpoints).

### Current Limitations & Future Work

### OCR Functionality: Currently disabled due to deployment complexity and dependency management.

We plan to integrate Tesseract OCR and Vision API support in future updates for full text extraction from scanned documents.


### Example Usage

Upload and Process a Document
python
Copy code
import requests

url = "http://127.0.0.1:8000/upload"
files = {"file": open("document.pdf", "rb")}
response = requests.post(url, files=files)
print(response.json())

### License

This project is licensed under the MIT License — see the LICENSE
 file for details.
