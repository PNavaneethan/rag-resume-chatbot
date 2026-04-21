import streamlit as st
import os
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# ---------------- UI CONFIGURATION ----------------
st.set_page_config(page_title="RAG Resume Chatbot", layout="wide")
st.title("📄 RAG Resume Chatbot")
st.write("Upload resumes and search for the best candidates using RAG.")

# ---------------- API KEY CHECK ----------------
# Note: While the code uses local models for embeddings and LLM, 
# you had a check for OpenAI. I've kept it as per your requirement.
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ Add OPENAI_API_KEY in Streamlit Secrets/Env")
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
            # Cleanup temp file
            os.remove(tmp_path)
            
        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {str(e)}")
            continue

    if not documents:
        st.error("❌ No valid PDF content found!")
        st.stop()

    # Split text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create Vector Store
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# ---------------- MAIN LOGIC ----------------
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} resumes uploaded successfully.")
    vector_db = process_pdfs(uploaded_files)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # Initialize Local LLM
    # We wrap the transformers pipeline in LangChain's HuggingFacePipeline
    hf_pipe = pipeline("text-generation", model="distilgpt2", max_new_tokens=200)
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    # Setup the RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever
    )

    query = st.text_input("🔍 Ask about candidates (e.g., 'Who has experience with Python?')")

    if query:
        with st.spinner("Searching resumes..."):
            try:
                # Using the chain for the response
                result = qa_chain.invoke(query)
                
                st.subheader("🧠 Answer")
                st.write(result["result"])
                
                # Show source documents in an expander for transparency
                with st.expander("View Source Content"):
                    source_docs = retriever.get_relevant_documents(query)
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.info(doc.page_content)
                        
            except Exception as e:
                st.error(f"An error occurred during retrieval: {e}")
