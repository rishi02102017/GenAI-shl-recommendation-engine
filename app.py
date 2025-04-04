import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

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
    df["embedding"] = df["combined_text"].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = load_data()

# UI
st.set_page_config(page_title="SHL Assessment Recommendation Engine")
st.title("🔍 SHL Assessment Recommendation System")
st.markdown("Enter a job description or hiring query below:")

query = st.text_area("Job Description / Query", height=200)
top_k = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Recommend"):
    if query.strip() == "":
        st.warning("Please enter a query or job description.")
    else:
        with st.spinner("Generating recommendations..."):
            query_embedding = model.encode(query, convert_to_tensor=True)
            scores = df["embedding"].apply(lambda x: util.cos_sim(query_embedding, x).item())
            df["score"] = scores
            results = df.sort_values("score", ascending=False).head(top_k)

            st.success(f"Top {top_k} recommended assessments:")
            st.dataframe(
                results[[
                    "Assessment Name", "URL", "Remote Testing",
                    "Adaptive/IRT", "Test Type"
                ]].reset_index(drop=True)
            )

            st.markdown("### 📋 Recommendations with Links")
            for idx, row in results.iterrows():
                st.markdown(f"""
**[{row['Assessment Name']}]({row['URL']})**

- **Remote Testing:** {row['Remote Testing']}
- **Adaptive/IRT:** {row['Adaptive/IRT']}
- **Test Type:** {row['Test Type']}
---
""")
