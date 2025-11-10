import streamlit as st
import pandas as pd
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFacePipeline # New LLM Import
from langchain.chains import RetrievalQA # New Chain Import
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline # New Hugging Face Imports


# --- CONFIGURATION ---
st.set_page_config(page_title="Local CSV RAG Tool", page_icon="🧠", layout="wide")
st.title("🧠 Healthcare Chatbot (Local RAG with FLAN-T5 + ChromaDB)")

# Define the path to your local CSV file
LOCAL_CSV_PATH = "data/drug.csv"
CHROMA_PATH = "chroma_db_store"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # For vectorizing data
LLM_MODEL_NAME = "google/flan-t5-small" # The LLM for generation (using the 'small' variant for faster testing)


# --- RAG COMPONENT FUNCTIONS ---

@st.cache_resource
def get_llm_chain(vector_db):
    """Loads FLAN-T5 and sets up the RetrievalQA chain."""
   
    st.info("4. Loading FLAN-T5 model (this may take a minute the first time)...")
    try:
        # Load Tokenizer and Model
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        # Use device_map='auto' to efficiently use CPU/GPU
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME, device_map="auto")
       
        # Create Hugging Face pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            truncation=True
        )
        llm = HuggingFacePipeline(pipeline=pipe)
       
        # Create the RAG Chain (RetrievalQA)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # Passes all context to the LLM at once
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}) # Use the top 3 results
        )
        st.success("✅ FLAN-T5 LLM and RAG Chain loaded successfully!")
        return qa_chain
    except Exception as e:
        st.error(f"🚨 Error loading FLAN-T5 or RAG chain: {e}")
        st.warning("Please ensure you have large model dependencies (`transformers`, `accelerate`, `bitsandbytes`) installed.")
        return None


@st.cache_resource
def create_vector_store(file_path):
    """Loads CSV, splits text, and creates a ChromaDB vector store."""
    try:
        st.info(f"1. Loading data from: **{file_path}**")
       
        # 1. Load the document
        # NOTE: You must update 'text_content_column' if you want a specific column indexed
        loader = CSVLoader(
            file_path=file_path,
            encoding="utf-8",
            csv_args={'delimiter': ','},
            # text_content_column='description' # Example: Uncomment and replace
        )
        documents = loader.load()
        st.info(f"Found **{len(documents)}** records in the CSV.")

        # 2. Split the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        st.info(f"2. Split into **{len(chunks)}** text chunks.")

        # 3. Load the local Sentence Transformer Embedding Model
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        st.info(f"3. Loaded local embedding model: **{EMBEDDING_MODEL_NAME}**")

        # 4. Create the ChromaDB Vector Store
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        db.persist()
        st.success("✅ ChromaDB Vector Store created and persisted successfully!")
       
        return db
       
    except FileNotFoundError:
        st.error(f"🚨 Error: CSV file not found at **{file_path}**. Please ensure the file exists!")
        return None
    except Exception as e:
        st.error(f"🚨 An error occurred during vectorization: {e}")
        return None

# --- MAIN APP LOGIC ---

# 1. Attempt to load and process the local CSV data
vector_db = create_vector_store(LOCAL_CSV_PATH)

# Initialize the RAG chain only if the vector store is ready
qa_chain = None
if vector_db:
    qa_chain = get_llm_chain(vector_db)


# --- QUERY LOGIC (Runs only if the RAG chain is successfully loaded) ---
if qa_chain:
    st.header("💬 Ask a Question (FLAN-T5 Generation)")

    user_query = st.text_input(
        'Enter your query (e.g., "Summarize the key findings for drug X?"):',
        value='',
        key="rag_query_input"
    )

    if st.button("Get Answer", type="primary") and user_query:
        try:
            with st.spinner("🧠 Querying RAG system and generating answer..."):
                # 5. Execute the RAG Chain to get the final answer
                result = qa_chain.invoke({"query": user_query})
           
            st.subheader("🤖 Chatbot Answer")
            st.info(result['result']) # The final generated answer
           
            # Optional: Show the chunks that were used to generate the answer
            with st.expander("Show Retrieved Context Chunks"):
                 # The RetrievalQA chain doesn't automatically return source documents with `run` or `invoke` unless configured.
                 # For simplicity, we can do a manual search to show context.
                 relevant_docs = vector_db.similarity_search(user_query, k=3)
                 for i, doc in enumerate(relevant_docs):
                    st.markdown(f"**Chunk {i+1}** (Source: `{doc.metadata.get('source', 'N/A')}`, Row: `{doc.metadata.get('row', 'N/A')}`)")
                    st.code(doc.page_content)
               
        except Exception as e:
            st.error(f"An error occurred during query generation: {e}")

# --- DISPLAY ORIGINAL DATAFRAME (Optional) ---
if os.path.exists(LOCAL_CSV_PATH):
    try:
        df = pd.read_csv(LOCAL_CSV_PATH)
        with st.expander("Show Original Data"):
            st.dataframe(df.head())
    except Exception:
        pass
