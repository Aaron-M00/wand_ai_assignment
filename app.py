import streamlit as st
import requests
import os

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Document Intelligence", page_icon="📄", layout="wide")

st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("Use the controls below to interact with the document intelligence system.")
mode = st.sidebar.radio("Select Mode", ["Upload Document", "Search / Q&A"])
st.sidebar.markdown("---")
st.sidebar.caption("Backend: FastAPI Service running at " + API_BASE_URL)

if mode == "Upload Document":
    st.title("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            with st.spinner(f"Uploading {file.name}..."):
                files = {"file": (file.name, file.getvalue(), file.type)}
                try:
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(f"{file.name} uploaded successfully.")
                    else:
                        st.error(f"Failed to upload {file.name}. Server responded with: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to FastAPI server. Make sure it’s running.")

elif mode == "Search / Q&A":
    st.title("🔍 Search or Question Answering")
    action = st.radio("Choose an action:", ["Search", "Q&A"])
    query = st.text_area("Enter your query:", placeholder="e.g. What is the summary of document X?")

    if st.button("Run Query"):
        if not query.strip():
            st.warning("Please enter a query first.")
        else:
            with st.spinner(f"Running {action.lower()}..."):
                try:
                    if action == "Search":
                        endpoint = "/search"
                        payload = {"query": query}
                    else:
                        endpoint = "/qa"
                        payload = {"question": query}

                    response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload)

                    if response.status_code == 200:
                        result = response.json()
                        st.subheader("🧠 Result")

                        # Pretty results
                        if action == "Search":
                            results = result.get("results", [])
                            if not results:
                                st.info("No relevant results found.")
                            else:
                                for i, r in enumerate(results, start=1):
                                    with st.container():
                                        st.markdown(
                                            f"""
                                            <div style="background-color:#f7f9fc;padding:15px;border-radius:12px;margin-bottom:10px;">
                                            <strong>Result {i}</strong><br>
                                            <span style="color:#333;">{r.get('page_content','')}</span>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                        else:
                            st.markdown(
                                f"""
                                <div style="background-color:#e8f0fe;padding:15px;border-radius:12px;margin-top:10px;">
                                <strong>Answer:</strong><br>
                                <span style="color:#222;">{result.get('answer', 'No answer returned.')}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    else:
                        st.error(f"Server returned error: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to FastAPI server. Please start it and try again.")

st.markdown("---")
st.caption("© 2025 Document Intelligence System | Powered by FastAPI + Streamlit")
