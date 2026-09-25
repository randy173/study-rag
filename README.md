# 📚 Study-RAG: Textbook AI Study Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-Inference%20Engine-F55036.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Study-RAG** is a high-performance, decoupled **Retrieval-Augmented Generation (RAG)** study companion designed specifically for heavy academic textbooks. It extracts multi-column PDFs, indexes them into a local vector database with zero-cost embeddings, and provides an interactive conversational assistant with verified, page-numbered citations.

---

## 🌟 Key Features

* **⚡ 100% Free Local Embeddings**: Built-in ONNX runtime (`all-MiniLM-L6-v2`) runs locally on your CPU. No OpenAI embedding fees, no billing accounts, and zero rate limits.
* **🚀 Blazing-Fast LLM Inference**: Native integration with [Groq](https://groq.com/) (`openai/gpt-oss-120b` or Llama) for instant streaming responses, with seamless fallback to OpenAI (`gpt-4o-mini`).
* **🧩 Adaptive Tolerance Chunking**: Eliminates awkward sentence splittings and orphan 1-word chunks with a dynamic [650–950 character] sentence-snapped window and tail remnant absorption.
* **🔍 Multi-Query Optimizer & Typo Normalizer**: Resolves student spelling mistakes (e.g., *"ecnomics"*), maps ordinals to numerals (*"7th principle"* ↔ *"Principle 7"*), strips diluting filler words, and retrieves across multiple search vectors in parallel.
* **📖 Verified Page Citations**: Every answer comes with expandable source drawers showing the exact textbook page number and relevant excerpts so you can double-check the material.
* **🛡️ Strict Anti-Hallucination Guardrails**: Prompts restrict answers solely to verified textbook context. If a concept isn't in your book, it explicitly states: *"This topic is not covered in your textbook."*
* **💾 Decoupled & Idempotent Architecture**: Batch PDF ingestion is completely separate from UI execution. Deterministic MD5 chunk IDs prevent duplicate embeddings and ensure instant vector loading.

---

## 🏛️ System Architecture

```
[Offline Ingestion Pipeline]
Textbook PDF in data/
   │
   ├─► PyMuPDF Block Extraction (Reading order sorted, headers/footers stripped)
   ├─► Adaptive Tolerance Chunker (Target 800 chars, ±150 window, sentence snapped)
   ├─► Idempotent MD5 Hasher (source:page:hash deduplication)
   └─► Local ChromaDB Vector Store (Persisted to chroma_db/)

[Interactive Streamlit App]
Student Question in Browser (http://localhost:8501)
   │
   ├─► Multi-Query Optimizer (Normalizes typos, converts ordinals, generates sub-queries)
   ├─► Deduplicated ChromaDB Retrieval (Top-k similarity matching across query variants)
   ├─► Contextual Document Synthesizer (Strict anti-hallucination prompt)
   └─► Groq / OpenAI LLM ──► Answer + Expandable Page Citations
```

---

## 🗂️ Project Structure

```
study-rag/
├── data/                      # Place your textbook PDFs here (e.g. 28074212.pdf)
│   └── .gitkeep               # Keeps folder structure in version control
├── chroma_db/                 # Local persistent Chroma vector database (ignored by git)
├── .env.example               # Template for API keys
├── .gitignore                 # Pre-configured to protect secrets and large PDFs
├── requirements.txt           # Pinned project dependencies
├── ingest.py                  # Offline batch ingestion script with adaptive chunking
├── app.py                     # Streamlit web application & conversational RAG UI
├── main.py                    # Convenience CLI dispatcher
├── diagnose_retrieval.py      # Vector retrieval test & diagnostic script
├── handoff.md                 # Technical architecture & engineering notes
└── README.md                  # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10+** installed on your system.
- Git installed.

### 2. Clone the Repository
```bash
git clone https://github.com/randy173/study-rag.git
cd study-rag
```

### 3. Install Dependencies
```powershell
# Windows
py -m pip install -r requirements.txt

# macOS / Linux
python3 -m pip install -r requirements.txt
```

### 4. Configure API Keys
Copy `.env.example` to `.env`:
```powershell
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` in your text editor and add your **Groq API Key** (Free at [console.groq.com](https://console.groq.com/keys)):
```env
GROQ_KEY=gsk_your_groq_api_key_here
```
*(Optional: If you prefer OpenAI, you can set `OPENAI_API_KEY=sk-...` instead).*

---

### 5. Ingest Your Textbook

1. Place your textbook PDF inside the `data/` folder (for example, `data/economics_textbook.pdf`).
2. Run the ingestion pipeline:

```powershell
# Windows
py ingest.py --reset

# macOS / Linux
python3 ingest.py --reset
```

> **Note:** The `--reset` flag clears any old indexes and cleanly creates the vector collection with the adaptive chunker. Subsequent runs without `--reset` are idempotent and will only process new or modified documents.

---

### 6. Launch the Application

You can launch both the **Flask Backend API** and the **React Modern Frontend**:

#### Option A: Quickstart via `main.py`
In two terminal tabs:
```powershell
# Tab 1: Start Flask Backend (Port 5000)
py main.py server

# Tab 2: Start React Frontend (Port 3000)
cd frontend
npm run dev
```

#### Option B: Production Single-Server Mode
Because the React app has been compiled into `frontend/dist/`, running `server.py` automatically serves the full React frontend and API together:
```powershell
py server.py
```
Then visit **[http://localhost:5000](http://localhost:5000)**!

---

## 🛠️ CLI Utilities & Commands

| Command | Description |
|---|---|
| `py ingest.py` | Ingests all PDFs in `data/` without wiping existing database entries. |
| `py ingest.py --reset` | Wipes existing `textbook_study` collection and performs a fresh adaptive re-indexing. |
| `py server.py` | Launches the Flask REST API (Port 5000) and serves compiled React frontend. |
| `cd frontend && npm run dev` | Launches the Vite React frontend with instant hot reload (Port 3000). |
| `py main.py server` | Dispatcher shortcut to run the Flask backend. |
| `py main.py frontend` | Dispatcher shortcut to start the React development server. |
| `py main.py ingest` | Dispatcher shortcut to run PDF ingestion. |
| `py diagnose_retrieval.py` | Diagnostic script to test query distance metrics and vector retrieval directly. |

---

## ⚙️ Configuration & Tech Stack

| Component | Technology | Role |
|---|---|---|
| **PDF Extraction** | PyMuPDF (`pymupdf`) | Block-level multi-column parsing with layout sorting |
| **Chunking Engine** | Adaptive Tolerance Chunker | Soft [650–950] char window, sentence snapping, tail absorption |
| **Embedding Engine** | ONNX `all-MiniLM-L6-v2` | Zero-cost, CPU-based local embeddings (or OpenAI fallback) |
| **Vector Store** | ChromaDB (`chromadb`) | Local persistent disk storage with cosine distance metrics |
| **LLM Inference** | Groq (`openai/gpt-oss-120b`) | Ultra-fast token generation with low latency |
| **Backend API** | Flask + Flask-CORS | High-throughput REST API with structured JSON endpoints |
| **Frontend UI** | React + Vite (Vanilla CSS) | Modern glassmorphism UI, citation drawers, query breakdowns |


---

## 🔒 Security & Privacy

- **API Keys**: All credentials are read from local environment variables (`.env`). The included `.gitignore` prevents `.env` and sensitive files from being pushed to public repositories.
- **Local Data**: Your textbook PDFs and ChromaDB vector indexes remain strictly on your local machine. Text is only sent to the LLM during query generation.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
