import streamlit as st
import pandas as pd
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

# LOAD MODEL 
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# UI 
st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
st.title("📊 Hybrid Multimodal RAG Assistant")
st.markdown("Supports CSV (structured) + PDF (unstructured) queries")

uploaded_file = st.file_uploader("Upload CSV or PDF", type=["csv", "pdf"])
question = st.text_input("Ask a question:")

# TEXT CHUNKING 
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# BUILD FAISS 
@st.cache_resource
def build_index(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

# SEARCH 
def search(query, chunks, index, k=5):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k=k)
    return [chunks[i] for i in I[0]]

# QUERY TYPE DETECTION 
def is_structured_query(question):
    keywords = ["highest", "lowest", "average", "sum", "top", "count", "max", "min"]
    return any(word in question.lower() for word in keywords)

# COLUMN DETECTION
def detect_column(question, df):
    q = question.lower()
    columns = df.columns.tolist()

    for col in columns:
        if col.lower() in q:
            return col

    for col in columns:
        if any(word in col.lower() for word in q.split()):
            return col

    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        return numeric_cols[0]

    return None

# PANDAS HANDLER 
def handle_with_pandas(df, question):
    q = question.lower()
    col = detect_column(question, df)

    if col is None:
        return "❌ Could not identify a relevant column."

    try:
        if "highest" in q or "max" in q or "top" in q:
            row = df.loc[df[col].idxmax()]
            return f"Highest {col}: {row[col]}\n\nRow Data:\n{row.to_dict()}"

        elif "lowest" in q or "min" in q:
            row = df.loc[df[col].idxmin()]
            return f"Lowest {col}: {row[col]}\n\nRow Data:\n{row.to_dict()}"

        elif "average" in q or "mean" in q:
            return f"Average {col}: {df[col].mean()}"

        elif "sum" in q:
            return f"Sum of {col}: {df[col].sum()}"

        elif "count" in q:
            return f"Total rows: {len(df)}"

        else:
            return None

    except Exception as e:
        return f"❌ Pandas error: {str(e)}"

# LLM GENERATION 
def generate_answer(question, context):
    context = context[:2000]

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the context below to answer the question.

If the answer is not clearly present, say:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = ollama.chat(
            model='llama3',
            messages=[{"role": "user", "content": prompt}]
        )

        return response.get('message', {}).get('content', "⚠️ Empty response")

    except Exception as e:
        return f"❌ Ollama Error: {str(e)}"

# CSV TO TEXT 
def dataframe_to_text(df):
    return "\n".join(
        df.astype(str).apply(lambda row: " | ".join(row), axis=1)
    )

# MAIN 
if uploaded_file is not None:

    file_type = uploaded_file.type

    # CSV
    if file_type == "text/csv":
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        if question and question.strip():

            # HYBRID LOGIC 
            if is_structured_query(question):
                st.info("⚡ Using Structured Engine (Pandas)")
                answer = handle_with_pandas(df, question)

                if answer:
                    st.success(answer)
                else:
                    st.warning("Fallback to RAG...")

                    text = dataframe_to_text(df)
                    chunks = chunk_text(text)
                    index = build_index(tuple(chunks))

                    results = search(question, chunks, index)
                    context = "\n".join(results)

                    with st.spinner("⏳ Generating answer..."):
                        answer = generate_answer(question, context)

                    st.success(answer)

            else:
                st.info("🔍 Using RAG Pipeline")

                text = dataframe_to_text(df)
                chunks = chunk_text(text)
                index = build_index(tuple(chunks))

                results = search(question, chunks, index)
                context = "\n".join(results)

                st.subheader("🔍 Retrieved Context")
                st.text(context[:500])

                with st.spinner("⏳ Generating answer..."):
                    answer = generate_answer(question, context)

                st.success(answer)

    # PDF
    elif file_type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            st.error("❌ Could not extract text from PDF.")
        else:
            st.subheader("📄 PDF Preview")
            st.text(text[:800])

            if question and question.strip():

                chunks = chunk_text(text)
                index = build_index(tuple(chunks))

                results = search(question, chunks, index)
                context = "\n".join(results)

                st.subheader("🔍 Retrieved Context")
                st.text(context[:500])

                with st.spinner("⏳ Generating answer..."):
                    answer = generate_answer(question, context)

                st.success(answer)

    else:
        st.error("Unsupported file type")
