import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Load CSV and compute embeddings
@st.cache_data
def load_data():
    df = pd.read_csv("shl_assessments.csv")
    df.fillna("Not Available", inplace=True)
    df["combined_text"] = (
        df["Assessment Name"].astype(str)
        + " "
        + df["Test Type"].astype(str)
        + " "
        + df["Remote Testing"].astype(str)
        + " "
        + df["Adaptive/IRT"].astype(str)
    )
    embeddings = model.encode(df["combined_text"].tolist(), convert_to_tensor=True)
    return df, embeddings

df, assessment_embeddings = load_data()

# Streamlit UI
st.set_page_config(page_title="SHL Assessment Recommendation Engine")
st.title("🔍 SHL Assessment Recommendation System")
st.markdown("Enter a job description or hiring query below:")

query = st.text_area("Job Description / Query", height=200)
top_k = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Recommend"):
    if query.strip() == "":
        st.warning("Please enter a query or job description.")
    else:
        with st.spinner("Finding recommendations..."):
            query_embedding = model.encode(query, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, assessment_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            results_df = df.iloc[top_results[1].cpu().numpy()]
            results_df = results_df.reset_index(drop=True)

            st.success(f"Top {top_k} recommended assessments:")
            st.dataframe(results_df[[
                "Assessment Name", "URL", "Remote Testing",
                "Adaptive/IRT", "Test Type"
            ]])

            st.markdown("### 📋 Recommendations with Links")
            for idx, row in results_df.iterrows():
                st.markdown(f"""
**[{row['Assessment Name']}]({row['URL']})**

- **Remote Testing:** {row['Remote Testing']}
- **Adaptive/IRT:** {row['Adaptive/IRT']}
- **Test Type:** {row['Test Type']}
---
""")
