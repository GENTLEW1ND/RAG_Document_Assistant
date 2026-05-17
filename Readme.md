# 📄 RAG Document Assistant (Hybrid Search + RRF + Reranking)

A **production-style Retrieval-Augmented Generation (RAG)** system that allows users to upload documents and ask intelligent questions using:

* 🔍 Hybrid Retrieval (Dense + Sparse)
* 🔄 Reciprocal Rank Fusion (RRF)
* 🎯 Cohere Reranking
* 🧠 LLM-based Answer Generation
* ⚡ Streamlit UI with session-based memory

---

# 🚀 Features

### 📌 Document Handling

* Upload any PDF document
* Automatic parsing, chunking, and summarization
* One-time processing per document (cached)

### 🔍 Advanced Retrieval Pipeline

* Multi-query generation for better recall
* Dense (vector) + Sparse (BM25) retrieval
* RRF to combine results robustly

### 🎯 Reranking

* Uses Cohere’s reranker to improve relevance
* Filters top candidates before LLM generation

### 💬 Interactive UI

* Ask multiple questions on the same document
* Upload a new document → automatic pipeline reset
* Debug view to inspect retrieved chunks

---

# 🧠 System Architecture

```text
User Query
    ↓
Query Rewriting (History-aware)
    ↓
Multi-Query Generation
    ↓
Dense Retrieval (Embeddings)
+ Sparse Retrieval (BM25)
    ↓
Reciprocal Rank Fusion (RRF)
    ↓
Top-K Selection
    ↓
Cohere Reranker
    ↓
LLM Answer Generation
    ↓
Response to User
```

---

# 📁 Project Structure

```text
rag_project/
│
├── Ingestion/
│   ├── LoadingandPartitioning.py
│   ├── ChunkingByTitle.py
│   ├── Unstructured_chunking.py
│   └── ConvertToEmbeddings.py
│
├── Retrival/
│   ├── MultiQuery.py
│   ├── DenseAndSparse.py
│   ├── Hybrid.py  (RRF implementation)
│   └── History.py
│
├── ReRanker/
│   └── Re_Ranker.py
│
├── Llm/
│   └── LlmGenerator.py
│
├── main.py        # Core pipeline logic
├── ui.py          # Streamlit UI
└── README.md
```

---

# ⚙️ Installation

```bash
git clone <your-repo-url>
cd rag_project

pip install -r requirements.txt
```

---

# 🔑 Environment Variables

You need a Cohere API key for reranking:

```bash
export COHERE_API_KEY="your_api_key_here"
```

---

# ▶️ Run the Application

```bash
streamlit run ui.py
```

---

# 🧪 How It Works

## 1. Upload Document

* File is hashed to detect changes
* Pipeline runs only for new documents
* Results are cached using Streamlit

## 2. Ask Questions

* Multiple queries supported per document
* No reprocessing required

## 3. Retrieval + Answering

* Multi-query improves recall
* RRF ensures robust ranking
* Reranker improves final relevance

---

# 🔥 Key Design Decisions

### ✅ Why RRF over Hybrid Retriever?

* Works better with multi-query pipelines
* Avoids score normalization issues
* More stable ranking across retrieval methods

### ✅ Why Not Use EnsembleRetriever?

* Limited control over scoring
* Less transparent for debugging
* RRF gives deterministic ranking behavior

### ✅ Why Session State?

* Streamlit reruns scripts on every interaction
* Session state preserves:

  * processed documents
  * embeddings
  * retrieval pipeline

---

# ⚡ Performance Optimizations

* `@st.cache_resource` for one-time document processing
* File hashing to prevent redundant computation
* Separation of setup vs query execution

---

# 🐞 Common Issues & Fixes

### ❌ Reranker Error (Invalid document type)

* Fix: Convert `Document` → `page_content` before sending to Cohere

### ❌ RRF Returning One Chunk

* Cause: Missing unique IDs
* Fix: Add `id` to document metadata

### ❌ Slow Performance

* Ensure caching is enabled
* Avoid re-running ingestion per query

---

# 🚀 Future Improvements

* 💬 Chat-style UI (conversation memory)
* 📊 Source highlighting in answers
* 🔎 Query/result visualization
* ⚙️ FastAPI backend + React frontend
* 🧠 Local reranker (to remove API dependency)

---

# 🧠 Key Learnings

* RAG quality depends more on **retrieval than generation**
* Multi-query + RRF significantly improves recall
* Reranking is essential for precision
* State management is critical in Streamlit apps

---

# 🤝 Contributing

Feel free to fork and improve this project. Suggestions and PRs are welcome!

---

# 📜 License

This project is open-source and available under the MIT License.
