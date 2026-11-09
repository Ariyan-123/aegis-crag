# 🛡️ AegisRAG: Enterprise Self-Corrective Hybrid Retrieval Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Qwen--27B-purple.svg)](https://groq.com/)

An autonomous, self-corrective Retrieval-Augmented Generation (CRAG) system engineered for enterprise documentation and runbooks. Mitigates retrieval failure and hallucinations via **Dense + Sparse Reciprocal Rank Fusion (RRF)**, **LangGraph cyclical routing**, and **deterministic relevance grading**.

---

## 🏛️ System Architecture

```text
                      +-----------------------------+
                      |     Enterprise Runbooks     |
                      |    (.pdf, .md, .txt)        |
                      +--------------+--------------+
                                     |
                          [ Recursive Chunking ]
                                     |
                 +-------------------+-------------------+
                 |                                       |
       [ Sparse Lexical ]                         [ Dense Vector ]
          BM25 Okapi                            all-MiniLM-L6-v2
                 |                                       |
                 +-------------------+-------------------+
                                     |
                       [ Reciprocal Rank Fusion ]
                               (k = 60)
                                     |
                        +------------v------------+
       +--------------->|   [LangGraph Engine]    |
       |                |     Retrieve Node       |
       |                +------------+------------+
       |                             |
       |                 [ Grade Documents Node ]
       |                 Relevance Verification
       |                             |
[Rewrite Query Node] <---- (Any Irrelevant?)
       |                             | (All Passed)
       +----------------<            v
                        +------------+------------+
                        |      Generate Node      |
                        |  Bounded Context Model  |
                        +------------+------------+
                                     |
                                     v
                        [ Grounded Synthesis ] 
```
---
## 🚀 Key Technical Highlights


- **Hybrid Retrieval (Dense + Sparse):** Pairs high-dimensional semantic search (`all-MiniLM-L6-v2` in Qdrant) with lexical exact-token matching (`BM25Okapi`) fused via Reciprocal Rank Fusion (RRF) at $k=60$.
- **Corrective Retrieval-Augmented Generation (CRAG):** Uses LangGraph to implement stateful cyclic graph logic. Documents pass through a binary relevance grader before generation; if relevance thresholds drop, an automated query transform rewrites the prompt and triggers a secondary retrieval cycle.
- **Zero-Hallucination Gating:** Strict bounding prevents models from answering out of parametric memory, returning clear remediation steps or triggering fallbacks.
- **Dual Presentation Interfaces:**
  - **Production REST API:** High-throughput async endpoints built with FastAPI and Pydantic validation schemas.
  - **Enterprise Observability Console:** Interactive Streamlit dashboard featuring round-trip latency, source chunk telemetry, and state-machine path traces.

---

## 📊 Quantitative Benchmark Evaluation

Evaluated against an enterprise runbook golden test suite across core RAG tri-factor metrics:

| Benchmark Question | Faithfulness | Answer Relevancy | Context Precision | Latency (s) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Remediation for code `ERR_BLK_403`** | **1.00** | **1.00** | **1.00** | 3.12s | ✅ Passed |
| **Default replication factor on data nodes** | **1.00** | **1.00** | **1.00** | 3.92s | ✅ Passed |
| **Rack awareness 3rd replica placement** | **1.00** | **1.00** | **1.00** | 8.41s | ✅ Passed |

*Summary: 100% faithfulness and zero hallucinations detected across all test iterations with full context retrieval accuracy. Output logged to `evaluation/benchmark_report.csv`.*

---

## 🛠️ Project Structure

```bash
aegis-crag/
├── data/
│   └── raw/                   # Raw enterprise policy & runbook files
├── evaluation/
│   └── benchmark_report.csv   # Auto-exported quantitative eval metrics
├── src/
│   ├── config.py              # Environment variables & hyperparameter configs
│   ├── api.py                 # FastAPI production endpoints
│   ├── evaluation/
│   │   └── ragas_eval.py      # Automated benchmarking & metric suite
│   ├── graph/
│   │   ├── nodes.py           # LangGraph node logic (retrieve, grade, rewrite, generate)
│   │   ├── state.py           # TypedDict GraphState definition
│   │   └── workflow.py        # Cyclic state graph assembly & compilation
│   ├── ingestion/
│   │   └── chunker.py         # Multi-format document loaders & semantic chunking
│   └── retrieval/
│       └── hybrid_search.py   # In-Memory Qdrant + BM25 fusion ranker
├── app.py                     # Streamlit enterprise observability UI
├── requirements.txt           # Production dependencies
└── README.md                  # System documentation
 
---

## ⚡ Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/<your-username>/aegis-crag.git
cd aegis-crag

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate 

pip install -r requirements.txt 
```

### 2. Configure Environment Keys
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Run Observability UI (Streamlit)
```bash
python -m streamlit run app.py 
```

### 4. Run REST API (FastAPI)
```bash
python -m uvicorn src.api:app --reload --port 8000
```

### 5. Execute Evaluation Benchmark Suite
```bash
python -m src.evaluation.ragas_eval
```