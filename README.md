
# 🔍 SHL Assessment Recommendation Engine

![Demo Banner](https://github.com/rishi02102017/shl-recommendation-engine/blob/main/SHL_logo.png)

A GenAI-powered tool to help hiring teams find the **most relevant SHL assessments** instantly based on **natural language queries**.

---

## 💡 Objective

Hiring managers often struggle to find the right assessments using keyword-based filters. Our goal was to **automate SHL test recommendations** using semantic search and language models.

---

## 🛠️ Tech Stack

- **Python + Pandas** – For backend logic and CSV preprocessing  
- **SentenceTransformers** – `all-MiniLM-L6-v2` for embedding queries & descriptions  
- **Streamlit** – Final UI for user interaction  
- **Gradio** – For exposing a temporary API endpoint that returns results in JSON  
- **Google Colab** – Used for development due to local system constraints  
- **GitHub** – Hosts all source code, dataset, and submission artifacts  

---

## 📂 Folder Structure

```
SHL/
├── app.py                         # ✅ Streamlit-based UI
├── api.py                         # ✅ FastAPI version (used via Gradio)
├── shl_assessments.csv           # ✅ Manually created from SHL Product Catalog
├── requirements.txt              # ✅ All dependencies for Streamlit + API
├── SHL_Final_Submission.ipynb    # ✅ Colab notebook with Gradio API
├── SHL_1-Page Approach Document.pdf  # ✅ Submission document
```

---

## 📊 Dataset

**Source**: [SHL Product Catalog](https://www.shl.com/solutions/products/product-catalog/)  
**Format**: Manually curated CSV  
**Rows**: ~20 assessments  
**Attributes**:
- Assessment Name  
- URL (SHL Link)  
- Remote Testing  
- Adaptive/IRT  
- Test Type  
- Duration

---

## ⚙️ How It Works

1. The query is semantically embedded using `all-MiniLM-L6-v2`
2. Each SHL assessment (concatenated name + type) is already embedded
3. We perform cosine similarity search to find top-k matches
4. Matches are shown in a table with all required fields

---

## 🚀 Live URLs

- **🧪 Working Demo (Streamlit)**:  
  https://shl-recommendation-engine.streamlit.app/

- **🧵 API Endpoint (via Gradio)**:  
  A temporary JSON-returning endpoint is available via Gradio.  
  ✅ Run the notebook [`SHL_Final_Submission.ipynb`](./SHL_Final_Submission.ipynb) in Colab  
  ✅ A new URL will be generated (valid for 72 hrs)

- **💻 GitHub Repo**:  
  https://github.com/rishi02102017/shl-recommendation-engine

---

## 🧠 Evaluation Metrics (Optional)

We computed the following based on sample ground-truth queries:

| Metric         | Value |
|----------------|--------|
| Recall@3       | 0.778  |
| MAP@3          | 0.778  |

---

## ⚙️ Hosting & API Attempts

We explored **multiple deployment methods** for hosting a persistent API, but faced consistent blockers:

| Platform     | Issue |
|--------------|-------|
| 🌀 **Ngrok**       | Tunnel instability + auth failures |
| 🔌 **LocalTunnel** | Blocked by college firewall |
| 🌐 **Render/Replit** | Disk limits or RAM crash |
| ✅ **Gradio**       | Used successfully as fallback via `share=True` |

---

## 📄 Approach Document

[Click to view the 1-page document](./SHL_1-Page%20Approach%20Document.pdf)

---

## 🧪 Sample Use Cases

- Looking for a **Java backend developer**?  
- Hiring **designers with Adobe tools**?  
- Need a **Data Analyst assessment under 30 minutes**?

Let this tool find the right tests — instantly.

---

## 📝 License

This repo is intended for the SHL GenAI Internship Assessment (2025).  
All content created solely for educational and assessment purposes.
