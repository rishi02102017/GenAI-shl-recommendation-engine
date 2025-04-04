import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

df = pd.read_csv("shl_assessments.csv")
df.columns = df.columns.str.strip()

# Convert row into a descriptive string
def row_to_text(row):
    return f"{row['Assessment Name']}. Remote Testing: {row['Remote Testing']}, Adaptive/IRT: {row['Adaptive/IRT']}, Test Type: {row['Test Type']}."

texts = df.apply(row_to_text, axis=1).tolist()

# Generate embeddings
@st.cache_data
def generate_embeddings(texts):
    return model.encode(texts, show_progress_bar=False)

embeddings = np.array(generate_embeddings(texts))

# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Streamlit UI
st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")
st.title(" SHL Assessment Recommendation System")
st.markdown("Enter a job description or query to get matching SHL assessments:")

query = st.text_area("🔍 Your Query", placeholder="e.g. Hiring for Python developer with SQL knowledge. Duration < 60 mins.")

if st.button("Recommend"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        q_embedding = model.encode([query])
        distances, indices = index.search(np.array(q_embedding), k=10)
        results = df.iloc[indices[0]].copy()
        results["Similarity Score"] = distances[0]
        st.success("Top SHL Assessments for your query:")
        st.dataframe(results[["Assessment Name", "URL", "Remote Testing", "Adaptive/IRT", "Test Type", "Similarity Score"]])
