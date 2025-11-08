import streamlit as st
import pandas as pd
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
# Try this if the first one fails after installation:
# from langchain.document_loaders.csv_loader import CSVLoader --check 1 --not required
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings # For local embeddings



# Assuming you might add LLM imports later, like this:
# from langchain_community.llms import HuggingFacePipeline


# --- CONFIGURATION ---
st.set_page_config(page_title="Local CSV RAG Tool", page_icon="🧠", layout="wide")
st.title("🧠 Healthcare Chatbot (Local RAG with ChromaDB)")

# Define the path to your local CSV file
# NOTE: Replace 'your_local_data.csv' with the actual path/name of your CSV file
LOCAL_CSV_PATH = "data/drug.csv" # Example path, assume a folder 'data' exists

# Define directories for storing data and vectors
CHROMA_PATH = "chroma_db_store"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # A lightweight, efficient local embedding model

# --- FUNCTION TO LOAD AND VECTORIZE DATA ---
@st.cache_resource
def create_vector_store(file_path):
    """Loads CSV, splits text, and creates a ChromaDB vector store."""
    try:
        st.info(f"1. Loading data from: **{file_path}**")
        
        # 1. Load the document using LangChain's CSVLoader
        # Specify the text_content_column - this is the column that contains the main text you want to embed.
        # If your data is structured, you might concatenate columns here, but for simplicity, we pick one.
        # NOTE: You must update 'main_text_column' to a relevant column in your CSV.
        loader = CSVLoader(
            file_path=file_path,
            encoding="utf-8",
            csv_args={
                'delimiter': ',',
            },
            # text_content_column='description' # Uncomment and replace 'description' with your main text column
        )
        documents = loader.load()
        
        st.info(f"Found **{len(documents)}** records in the CSV.")

        # 2. Split the documents into smaller chunks (if necessary for larger files)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
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
        db.persist() # Save the vector store to disk
        st.success("✅ ChromaDB Vector Store created and persisted successfully!")
        
        return db, embeddings
        
    except FileNotFoundError:
        st.error(f"🚨 Error: CSV file not found at **{file_path}**. Please ensure the file exists!")
        return None, None
    except Exception as e:
        st.error(f"🚨 An error occurred during vectorization: {e}")
        st.warning("Please ensure you have all required libraries (like `langchain`, `sentence-transformers`, `chromadb`) installed.")
        return None, None

# --- MAIN APP LOGIC ---

# 1. Attempt to load and process the local CSV data
vector_db, embeddings_model = create_vector_store(LOCAL_CSV_PATH)

# --- QUERY LOGIC (Runs only if vector store is created) ---
if vector_db and embeddings_model:
    st.header("Ask a Question (Vector Search)")

    query_text = st.text_input(
        'Enter your query (e.g., "What were the side effects for drug X?"):',
        value='',
        key="rag_query_input"
    )

    # 2. Perform Similarity Search
    if query_text:
        try:
            with st.spinner("Searching ChromaDB..."):
                # Use similarity search to find the most relevant document chunks
                # k=3 retrieves the top 3 most relevant results
                relevant_docs = vector_db.similarity_search(query_text, k=3)
            
            st.subheader("Retrieval Results (Relevant Chunks)")
            
            # Display the relevant document chunks
            if relevant_docs:
                for i, doc in enumerate(relevant_docs):
                    st.markdown(f"**Result {i+1}** (Source: `{doc.metadata.get('source', 'N/A')}`, Row: `{doc.metadata.get('row', 'N/A')}`)")
                    st.code(doc.page_content)
                
                # NOTE: The next step in a full RAG app would be to take these chunks
                # and pass them to an LLM (like FLAN-T5) to generate a natural language answer.
                st.info("💡 **Next Step:** Integrate an LLM (e.g., FLAN-T5) to generate a cohesive answer from these retrieved chunks.")
            else:
                st.warning("No relevant chunks found for this query.")
                
        except Exception as e:
            st.error(f"An error occurred during search: {e}")

# --- DISPLAY ORIGINAL DATAFRAME (Optional) ---
# To still display the original data frame, we can load it separately
if os.path.exists(LOCAL_CSV_PATH):
    try:
        df = pd.read_csv(LOCAL_CSV_PATH)
        with st.expander("Show Original Data"):
            st.dataframe(df.head())
    except Exception:
        pass # Ignore errors if file is corrupted