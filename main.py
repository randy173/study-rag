"""
study-rag Entry Point Helper

The application is decoupled into two dedicated modules:
1. Ingestion: Processes PDFs, chunks text, and builds the local ChromaDB vector store.
2. Web UI: Launches the Streamlit study assistant interface.

Usage:
  - Ingest textbook PDFs:
      py ingest.py

  - Launch interactive UI:
      py -m streamlit run app.py
"""

import sys
import subprocess

def main():
    print("=" * 60)
    print("  📚 study-rag: Textbook AI Assistant")
    print("=" * 60)
    print("Available commands:")
    print("  1. Ingest PDFs in data/  -> run: py ingest.py")
    print("  2. Start Web Interface   -> run: py -m streamlit run app.py")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        subprocess.run([sys.executable, "ingest.py"])
    elif len(sys.argv) > 1 and sys.argv[1] == "app":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
