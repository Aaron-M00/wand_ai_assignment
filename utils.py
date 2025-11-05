import os
import json
import hashlib
import shutil
from typing import List
from datetime import datetime
from tqdm import tqdm

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)


def check_ocr_dependencies():
    missing = []
    for dep in ["tesseract", "pdftotext"]:
        if shutil.which(dep) is None:
            missing.append(dep)
    if missing:
        print(
            f"Missing OCR dependencies: {', '.join(missing)}. "
            "Install them with: brew install poppler tesseract"
        )


check_ocr_dependencies()


def compute_file_hash(path: str) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_documents(directory_path: str):
    supported_exts = (".pdf", ".txt", ".csv", ".docx")
    all_docs = []
    files = [f for f in os.listdir(directory_path) if f.lower().endswith(supported_exts)]

    if not files:
        print("No supported files found in directory.")
        return []

    print(f"Found {len(files)} supported files. Starting ingestion...\n")

    for file_name in tqdm(files, desc="Ingesting documents", unit="file"):
        file_path = os.path.join(directory_path, file_name)
        try:
            if file_name.lower().endswith(".txt"):
                loader = TextLoader(file_path)
            elif file_name.lower().endswith(".csv"):
                loader = CSVLoader(file_path)
            elif file_name.lower().endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_name.lower().endswith(".pdf"):
                loader = UnstructuredFileLoader(file_path)
            else:
                continue
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    print(f"Ingestion complete. Loaded {len(all_docs)} documents.")
    return all_docs


def chunk_documents(docs: List[Document], chunk_size=2000, overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )
    return text_splitter.split_documents(docs)


def save_metadata(metadata_path: str, data: dict):
    with open(metadata_path, "w") as f:
        json.dump(data, f, indent=2)


def load_metadata(metadata_path: str) -> dict:
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}


def create_or_load_vector_store(persist_directory: str, docs=None):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def sanitize_metadata(doc):
        clean_meta = {}
        for k, v in (getattr(doc, "metadata", {}) or {}).items():
            if isinstance(v, (list, dict)):
                clean_meta[k] = json.dumps(v)
            elif isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        doc.metadata = clean_meta
        return doc

    if docs:
        docs = [sanitize_metadata(d) for d in docs if getattr(d, "page_content", "").strip()]
        if not docs:
            print("No valid text to embed — skipping update.")
            return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        chunked_docs = text_splitter.split_documents(docs)
        print(f"Embedding {len(chunked_docs)} document chunks...")

        vectordb = Chroma.from_documents(
            chunked_docs,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        vectordb.persist()
        print("Embedding complete and database persisted.")
        return vectordb

    if os.path.exists(persist_directory):
        print("Loading existing Chroma DB...")
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
    else:
        print(f"Creating new Chroma DB in {persist_directory} ...")
        os.makedirs(persist_directory, exist_ok=True)
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )

    try:
        count = vectordb._collection.count()
        print(f"Chroma store ready with {count} documents.")
    except Exception:
        print("Could not count documents, but Chroma store initialized.")

    return vectordb


def incremental_update(docs_path: str, persist_directory: str):
    metadata_path = os.path.join(persist_directory, "metadata.json")
    previous_state = load_metadata(metadata_path)
    current_state = {}

    for root, _, files in os.walk(docs_path):
        for f in files:
            file_path = os.path.join(root, f)
            if f.lower().endswith((".pdf", ".txt", ".md", ".docx")):
                current_state[file_path] = compute_file_hash(file_path)

    added_or_changed = [
        f for f, h in current_state.items()
        if f not in previous_state or previous_state[f] != h
    ]
    deleted_files = [f for f in previous_state if f not in current_state]

    print(f"Added/Updated: {len(added_or_changed)}, Deleted: {len(deleted_files)}")

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)

    docs = load_documents(docs_path)
    docs = chunk_documents(docs)
    vectordb = create_or_load_vector_store(persist_directory, docs)
    save_metadata(metadata_path, current_state)
    print("Incremental sync complete.")
    return vectordb


def semantic_search(vectordb, query: str, k: int = 5):
    results = vectordb.similarity_search(query, k=k)
    print(f"Found {len(results)} relevant results.")
    return results


def generate_answer(llm, vectordb, query: str):
    results = semantic_search(vectordb, query)
    context = "\n".join([r.page_content for r in results])

    prompt = (
        f"Use the following context to answer accurately. "
        f"If you don't know, say 'I don’t know.'\n\nContext:\n{context}\n\nQ: {query}\nA:"
    )

    response = llm.invoke(prompt)
    print(response.content.strip())
    return response.content.strip()


def check_completeness(vectordb, docs_path: str) -> dict:
    if not isinstance(docs_path, str) or not os.path.isdir(docs_path):
        raise ValueError(f"Expected a directory path for docs_path, got: {docs_path}")

    local_files = [
        f for f in os.listdir(docs_path)
        if os.path.isfile(os.path.join(docs_path, f))
    ]
    num_local_files = len(local_files)

    try:
        vectordb_count = vectordb._collection.count()
    except Exception:
        vectordb_count = len(vectordb.get().get("ids", []))

    status = "Complete" if vectordb_count >= num_local_files else "Incomplete"

    print(
        f"[Completeness Check] {status} | "
        f"Local files: {num_local_files}, Indexed vectors: {vectordb_count}"
    )

    return {
        "status": status,
        "local_docs": num_local_files,
        "indexed_vectors": vectordb_count,
    }
