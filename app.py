import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

#  Page Config
st.set_page_config(page_title="SHL Assessment Recommendation Engine", layout="centered")
st.title("🔍 SHL Assessment Recommendation Engine")
st.markdown(
    "<span style='color:gray'>Helping you match SHL tests to your job roles in seconds 🚀</span>",
    unsafe_allow_html=True
)


#  Load SHL assessments
df = pd.read_csv("shl_assessments.csv")
df.columns = df.columns.str.strip()  # Strip column names
df.fillna("Not Available", inplace=True)

#  Create a richer field for embedding
df["Full Description"] = df["Assessment Name"] + " " + df["Test Type"]

#  Load model and cache embeddings
@st.cache_data
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["Full Description"].tolist(), convert_to_tensor=True)
    return model, embeddings

model, corpus_embeddings = load_embeddings()

#  Sidebar or example prompts
with st.expander(" Example Queries"):
    st.markdown("""
    - Looking for a **Java backend developer**  
    - Hiring a **DevOps engineer** with AWS and Docker  
    - Need someone with **UI/UX skills** using Adobe tools  
    - Looking for a **Data Analyst** familiar with Excel  
    """)
    st.divider()

#  User Input
user_query = st.text_area("📋 Paste a job description or skill query", height=150)

# 🎚 Slider & filters
st.subheader(" How many recommendations do you want?")
top_k = st.slider("", min_value=1, max_value=10, value=5)

remote_only = st.checkbox(" Remote-enabled only", help="Only show assessments that support remote testing")
simulation_only = st.checkbox(" Simulation-type only (S)", help="Only show assessments labeled as type S")

#  Trigger Button
if st.button(" Recommend Assessments") and user_query.strip():
    with st.spinner("Generating smart recommendations..."):
        query_embedding = model.encode(user_query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

        results = []
        for hit in hits:
            i = hit["corpus_id"]
            row = df.iloc[i]
            result = {
                 "Assessment Name": f"[{row['Assessment Name']}]({row['URL']})",
                 "Remote Testing": row["Remote Testing"],
                 "Adaptive/IRT": row["Adaptive/IRT"],
                 "Duration": row["Duration"],  
                 "Test Type": row["Test Type"],
                 "Score": round(hit["score"], 4),
             }


            #  Apply filters
            if remote_only and result["Remote Testing"].lower() != "yes":
                continue
            if simulation_only and "S" not in result["Test Type"]:
                continue

            results.append(result)

        #  Final Display
        if results:
            st.divider()
            st.subheader("🔎 Top Recommendations")
            st.dataframe(pd.DataFrame(results).drop(columns=["Score"]))
        else:
            st.warning("⚠️ No results match your filters. Try changing the filters or modifying the job description.")
