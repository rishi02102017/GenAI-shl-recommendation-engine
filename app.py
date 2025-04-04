import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import json

# Load SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Load SHL assessment metadata (update this to your actual path or source)
@st.cache_data
def load_assessment_data():
    df = pd.read_csv("shl_assessments.csv")
    return df

df = load_assessment_data()

# Generate embeddings for assessments (on description or combined fields)
@st.cache_data
def generate_embeddings(df):
    texts = df["description"].fillna("").tolist()  # or combine multiple fields if needed
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)

assessment_embeddings = generate_embeddings(df)

# Streamlit UI
st.title("🔍 SHL Assessment Recommendation Engine")
query = st.text_area("Enter job description or query here:", height=200)

if st.button("Recommend Assessments"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Generating recommendations..."):
            query_embedding = model.encode([query])
            similarities = cosine_similarity(query_embedding, assessment_embeddings)[0]
            top_k = 10
            indices = similarities.argsort()[-top_k:][::-1]

            # Prepare and display recommendations
            results = []
            for i in indices:
                row = df.iloc[i]
                results.append({
                    "Assessment Name": f"[{row['name']}]({row['url']})",
                    "Remote Testing Support": row.get("remote_support", "Unknown"),
                    "Adaptive/IRT Support": row.get("adaptive_support", "Unknown"),
                    "Duration": row.get("duration", "N/A"),
                    "Test Type": row.get("test_type", "N/A"),
                    "Relevance Score": f"{similarities[i]:.2f}"
                })

            results_df = pd.DataFrame(results)
            st.markdown("### 📋 Top Recommendations")
            st.dataframe(results_df, use_container_width=True)
