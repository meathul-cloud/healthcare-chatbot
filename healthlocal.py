import streamlit as st
import pandas as pd
import os # Keep os for file handling

st.set_page_config(page_title="Local CSV Query Tool", page_icon="📊", layout="wide")
st.title("📊 Healthcare Chatbot (Local RAG with FLAN-T5 + ChromaDB)")

# --- FILE UPLOAD (Modified for CSV) ---
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
df = None # Initialize DataFrame

if uploaded_file is not None:
    try:
        # 1. Read the CSV file directly into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV file loaded successfully!")
        st.subheader("First 5 Rows of Data:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# --- INPUT BOX AND QUERY LOGIC (Runs only if data is loaded) ---
if df is not None:
    st.header("Ask a Question (Simple Filtering)")

    # 2. Input Box for filtering - Using column selection for structure
    col_to_filter = st.selectbox(
        'Select a Column:',
        options=df.columns
    )

    query_value = st.text_input(
        f'Enter value to search for in "{col_to_filter}":',
        value='',
        key="query_input"
    )

    # 3. Populate the result
    if query_value:
        try:
            # Simple case-insensitive string containment filter
            filtered_df = df[df[col_to_filter].astype(str).str.contains(query_value, case=False, na=False)]
            
            st.subheader("Query Results")
            if not filtered_df.empty:
                st.dataframe(filtered_df)
                st.info(f"Found {len(filtered_df)} matching rows.")
            else:
                st.warning("No matching records found.")
        except Exception as e:
            st.error(f"An error occurred during filtering: {e}")