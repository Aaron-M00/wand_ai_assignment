# Document Intelligence System

## Overview

The Document Intelligence System is a modular platform for document ingestion, semantic search, and question answering. It integrates a FastAPI backend for processing and reasoning with a Streamlit frontend for interactive document exploration. Users can upload documents, perform semantic searches, and ask context-grounded questions. The system leverages modern NLP, vector databases, and LLMs for intelligent document understanding and retrieval.

---

## Architecture

📂 Project Root
├── main.py # Core ingestion and QA pipeline logic
├── utils.py # Helper utilities for vector store and search
├── server.py # FastAPI backend exposing upload, search, and QA APIs
├── app.py # Streamlit frontend for document interaction
├── requirements.txt # Dependencies list
├── .env # Environment variables (API keys and configuration)
└── README.md # Documentation

yaml
Copy code

---

## Features

- Document upload through the Streamlit interface  
- Automatic ingestion and background vectorization of uploaded documents  
- Semantic search using ChromaDB  
- Question answering over embedded document context  
- RESTful backend powered by FastAPI  
- Streamlit frontend with clean, chat-style results  
- Background ingestion for smooth user experience  

---

## Design Decisions

### Multi-Tier Architecture
The system is divided into two independent components:
- **FastAPI backend** for processing and reasoning
- **Streamlit frontend** for user interaction

This separation improves modularity, scalability, and maintainability.

### Asynchronous Ingestion
After document upload, ingestion is triggered in the background, allowing users to continue interacting with the system without delay.

### Vector Store
Documents are embedded and stored in **ChromaDB**, enabling fast retrieval for both search and question answering.

### LangChain Integration
LangChain orchestrates document parsing, embedding, and retrieval workflows, simplifying the pipeline and model coordination.

### Streamlit Frontend
The Streamlit interface connects to the backend via HTTP requests and renders search and QA responses in conversational bubbles for a clear and readable presentation.

---

## API Endpoints

| Endpoint | Method | Description |
|-----------|--------|-------------|
| `/` | GET | Health check |
| `/upload` | POST | Upload a document and trigger ingestion |
| `/ingest` | POST | Start ingestion manually |
| `/search` | POST | Perform semantic search with `{"query": "..."}` |
| `/qa` | POST | Perform question answering with `{"question": "..."}` |

---

## Environment Configuration

Before running the system, create a `.env` file in the project root to securely store API credentials and environment variables.

### Example `.env` File

OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
CHROMA_DB_PATH=./chroma_store

php
Copy code

If you use any other APIs (e.g., Google Vision or Hugging Face), include them here as well:

GOOGLE_APPLICATION_CREDENTIALS=path_to_your_google_service_account.json
HUGGINGFACE_API_KEY=your_huggingface_token_here

yaml
Copy code

Make sure the `.env` file is **not committed to version control** by adding it to `.gitignore`.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
2. Create and Activate Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Configure Environment Variables
Create the .env file as described above and add your API keys.

5. Start the FastAPI Backend
bash
Copy code
python server.py
The backend will be available at:

arduino
Copy code
http://localhost:8000
6. Launch the Streamlit Frontend
bash
Copy code
streamlit run app.py
The frontend will run at:

arduino
Copy code
http://localhost:8501
Usage Workflow
Start the FastAPI backend.

Launch the Streamlit interface.

Upload documents via the Upload Document tab.

Once uploaded, ingestion starts automatically in the background.

Switch to the Search / Q&A tab to perform queries or ask questions.

View results displayed in styled response bubbles for readability.

Future Enhancements
OCR-based image PDF integration

Role-based user authentication

Cloud-based vector storage and deployment support

Conversation memory for persistent query context

Document analytics and summary visualization

License
This project is distributed under the MIT License.
