import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load SHL assessments
df = pd.read_csv("shl_assessments.csv")

# Strip column names to avoid KeyErrors
df.columns = df.columns.str.strip()

# Fill NaN values
df.fillna("Not Available", inplace=True)

# Create a new field to embed richer info
df["Full Description"] = df["Assessment Name"] + " " + df["Test Type"]

@st.cache_data
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = model.encode(df["Full Description"].tolist(), convert_to_tensor=True)
    return model, corpus_embeddings

model, corpus_embeddings = load_embeddings()

# UI
st.title("🔍 SHL Assessment Recommendation Engine")
user_query = st.text_area("Paste a job description or query here", height=150)
top_k = st.slider("How many recommendations do you want?", min_value=1, max_value=10, value=5)

if st.button("Recommend Assessments") and user_query.strip():
    with st.spinner("Generating recommendations..."):
        query_embedding = model.encode(user_query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

        results = []
        for hit in hits:
            i = hit["corpus_id"]
            score = hit["score"]
            row = df.iloc[i]
            results.append({
                "Assessment Name": f"[{row['Assessment Name']}]({row['URL']})",
                "Remote Testing": row["Remote Testing"],
                "Adaptive/IRT": row["Adaptive/IRT"],
                "Test Type": row["Test Type"],
                "Score": round(score, 4),
            })

        st.markdown("### 🔎 Top Recommendations")
        st.dataframe(pd.DataFrame(results).drop(columns=["Score"]))
