"""
study-rag Entry Point Helper

The application is decoupled into:
1. Ingestion: Processes PDFs, chunks text, and builds the local ChromaDB vector store.
2. Flask Backend: REST API serving RAG vector search, query optimization, and LLM inference.
3. React Frontend: Interactive modern study UI with live citations.

Usage:
  - Ingest textbook PDFs:
      py ingest.py
  - Launch Flask Backend API (Port 5000):
      py server.py
  - Launch React Frontend (Port 3000):
      cd frontend && npm run dev
"""

import sys
import subprocess

def main():
    print("=" * 60)
    print("  [Study-RAG] Textbook AI Assistant (Flask + React)")
    print("=" * 60)
    print("Available commands:")
    print("  1. Ingest PDFs in data/     -> py ingest.py")
    print("  2. Start Flask Backend API  -> py server.py")
    print("  3. Start React Frontend     -> cd frontend; npm run dev")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "ingest":
            subprocess.run([sys.executable, "ingest.py"])
        elif cmd == "server":
            subprocess.run([sys.executable, "server.py"])
        elif cmd == "app":
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        elif cmd in ("frontend", "ui"):
            subprocess.run(["npm", "run", "dev"], cwd="frontend", shell=True)

if __name__ == "__main__":
    main()

