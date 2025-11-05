import os
import threading
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence

from utils import (
    incremental_update,
    create_or_load_vector_store,
    check_completeness,
    semantic_search,
)

load_dotenv()
PERSIST_DIR = "./chroma_store"
DOCS_PATH = "./docs"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY environment variable.")


def run_ingestion() -> None:
    print("Running document sync...")
    vectordb = incremental_update(DOCS_PATH, PERSIST_DIR)
    check_completeness(vectordb, DOCS_PATH)


def run_semantic_search(query: str, k: int = 5):
    vectordb = create_or_load_vector_store(PERSIST_DIR)
    results = semantic_search(vectordb, query, k=k)

    if not results:
        print("No results found.")
        return

    print("\nTop Search Results")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.page_content[:300]}...")
    print("\nEnd of results.")


def qa_pipeline(query: str) -> str:
    vectordb = create_or_load_vector_store(PERSIST_DIR)
    retriever = vectordb.as_retriever(search_kwargs={"k": 50})
    results = retriever.invoke(query)
    if not results:
        return "I don’t know."

    print("\nRetrieved Context Snippets")
    for i, r in enumerate(results[:2], 1):
        snippet = r.page_content[:400].replace("\n", " ")
        print(f"\n[{i}] {snippet}...\n")

    context = "\n\n---\n\n".join(r.page_content for r in results)

    prompt = ChatPromptTemplate.from_template("""
Use ONLY the provided context to answer the question accurately and completely.
If the context implies the answer indirectly, infer it using reasoning.
If the answer is missing, say "I don’t know."

Context:
{context}

Question: {question}

Answer:
""")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    chain = RunnableSequence(prompt | llm)
    response = chain.invoke({"context": context, "question": query}).content.strip()

    if response.lower().startswith("i don’t know") or len(response) < 40:
        print("Expanding context (k=80)...")
        retriever = vectordb.as_retriever(search_kwargs={"k": 80})
        results = retriever.invoke(query)
        context = "\n\n---\n\n".join(r.page_content for r in results)
        response = chain.invoke({"context": context, "question": query}).content.strip()

    print(response)
    return response


def run_qa(query: str, persist_dir: str = PERSIST_DIR) -> str:
    return qa_pipeline(query)


def run_local_interface():
    while True:
        print("\nAI Hybrid Workforce – Local CLI Interface")
        print("==============================================")
        print("Options:")
        print("1. Ingest / Sync Documents")
        print("2. Semantic Search")
        print("3. Q&A")
        print("4. Exit\n")

        choice = input("Enter choice (1–4): ").strip()

        if choice == "1":
            print("\nStarting ingestion...")
            thread = threading.Thread(target=run_ingestion)
            thread.start()
            thread.join()
            print("\nIngestion complete. Returning to main menu...")

        elif choice == "2":
            while True:
                query = input("\nEnter your search query (or 'back' to return): ").strip()
                if query.lower() == "back":
                    break
                k_input = input("How many results to show? (default 5): ").strip()
                k = int(k_input) if k_input.isdigit() else 5
                run_semantic_search(query, k)

        elif choice == "3":
            while True:
                question = input("\nAsk a question (or 'back' to return): ").strip()
                if question.lower() == "back":
                    break
                run_qa(question)

        elif choice == "4":
            print("\nExiting... Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1–4.")


if __name__ == "__main__":
    run_local_interface()
