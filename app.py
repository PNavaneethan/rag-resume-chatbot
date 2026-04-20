import streamlit as st
import os
from tempfile import NamedTemporaryFile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ---------------- UI ----------------
st.set_page_config(page_title="RAG Resume Chatbot")
st.title("📄 RAG Resume Chatbot")
st.write("Upload multiple resumes and ask about candidates")

# ---------------- API KEY ----------------
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OpenAI API key not found. Add it in Streamlit Secrets.")
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
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            documents.extend(loader.load())

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
    st.success(f"{len(uploaded_files)} resumes uploaded successfully!")

    vector_db = process_pdfs(uploaded_files)

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=openai_api_key
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """You are an HR assistant.

Use the context below to answer the question about candidates.

Context:
{context}

Question:
{input}

Answer clearly with candidate details.
"""
    )

    # Chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # User Query
    query = st.text_input("🔍 Ask about candidates")

    if query:
        with st.spinner("Searching resumes..."):
            response = retrieval_chain.invoke({"input": query})

            st.subheader("🧠 Answer")
            st.write(response["answer"])
