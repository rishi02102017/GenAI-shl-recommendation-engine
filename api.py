from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import pandas as pd
import uvicorn
from sentence_transformers import SentenceTransformer, util

# Step 1: Start FastAPI app
app = FastAPI(title="SHL Recommender API")

# Step 2: Load data and model
df = pd.read_csv("shl_assessments.csv")
df.fillna("Not Available", inplace=True)
df["Full Description"] = df["Assessment Name"] + " " + df["Test Type"]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["Full Description"].tolist(), convert_to_tensor=True)

# Step 3: API Response Schema
class Assessment(BaseModel):
    assessment_name: str
    url: str
    remote_testing: str
    adaptive_irt: str
    test_type: str
    duration: str

# Step 4: Define route
@app.get("/recommend", response_model=List[Assessment])
def recommend(query: str = Query(..., description="Job role or query"), top_k: int = 5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]

    results = []
    for hit in hits:
        i = hit["corpus_id"]
        row = df.iloc[i]
        result = {
            "assessment_name": row["Assessment Name"],
            "url": row["URL"],
            "remote_testing": row["Remote Testing"],
            "adaptive_irt": row["Adaptive/IRT"],
            "test_type": row["Test Type"],
            "duration": row["Duration"]
            
        }
        results.append(result)
    return results

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=10000)

