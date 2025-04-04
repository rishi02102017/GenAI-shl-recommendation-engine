import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# === Load Data ===
df = pd.read_csv("shl_assessments.csv")

# Clean column names and fill missing values
df.columns = df.columns.str.strip()
df.fillna("Not Available", inplace=True)

# Combine fields to create richer embedding context
df["Full Description"] = (
    df["Assessment Name"].astype(str) + " " +
    df["Test Type"].astype(str) + " " +
    df["Remote Testing"].astype(str) + " " +
    df["Adaptive/IRT"].astype(str)
)

# === Load Model & Embeddings (cached) ===
@st.cache_data
def load_model_and_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = model.encode(
        df["Full Description"].tolist(), convert_to_tensor=True
    )
    return model, corpus_embeddings

model, corpus_embeddings = load_model_and_embeddings()

# === Streamlit UI ===
st.set_page_config(page_title="SHL Recommendation Engine", layout="centered")
st.title("🔍 SHL Assessment Recommendation Engine")

st.markdown("Paste a **job description or skill query** below to get tailored SHL test suggestions:")

user_query = st.text_area("📝 Job Description or Role Query", height=150)
top_k = st.slider("📌 How many recommendations do you want?", 1, 10, 5)

# === Recommend Logic ===
if st.button("🚀 Recommend Assessments") and user_query.strip():
    with st.spinner("Thinking hard... 🧠"):
        query_embedding = model.encode(user_query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

        recommendations = []
        for hit in hits:
            idx = hit["corpus_id"]
            row = df.iloc[idx]
            recommendations.append({
                "Assessment Name": f"[{row['Assessment Name']}]({row['URL']})",
                "Remote Testing": row["Remote Testing"],
                "Adaptive/IRT": row["Adaptive/IRT"],
                "Test Type": row["Test Type"],
                "Score": round(hit["score"], 4),  # optional debug
            })

        st.markdown("### 🔎 Top Recommendations")
        st.dataframe(pd.DataFrame(recommendations).drop(columns=["Score"]))

# === Footer ===
st.markdown("---")
st.caption("Built with ❤️ using Sentence Transformers and Streamlit")
