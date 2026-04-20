import streamlit as st
import os
from tempfile import NamedTemporaryFile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA  # ✅ Stable import

# ---------------- UI ----------------
st.set_page_config(page_title="RAG Resume Chatbot")
st.title("📄 RAG Resume Chatbot")
st.write("Upload resumes and search candidates")

# ---------------- API KEY ----------------
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ Add OPENAI_API_KEY in Streamlit Secrets")
    st.stop()

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------- PROCESS PDFs ----------------
@st.cache_resource
def process_pdfs(files):
    documents = []

    for file in files:
        try:
            if file.size == 0:
                st.warning(f"⚠️ Skipping empty file: {file.name}")
                continue

            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            if not docs:
                st.warning(f"⚠️ No readable content in: {file.name}")
                continue

            documents.extend(docs)

        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {str(e)}")
            continue

    if not documents:
        st.error("❌ No valid PDF content found!")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db
# ---------------- MAIN ----------------
if uploaded_files:
    st.success(f"{len(uploaded_files)} resumes uploaded")

    vector_db = process_pdfs(uploaded_files)

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=openai_api_key
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    query = st.text_input("🔍 Ask about candidates")

    if query:
        with st.spinner("Searching..."):
            result = qa_chain.run(query)

            st.subheader("🧠 Answer")
            st.write(result)
