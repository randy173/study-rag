import os
import re
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# LangChain & ChromaDB
import chromadb
import chromadb.utils.embedding_functions as ef
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for local React development (typically port 5173 for Vite or 3000)
CORS(app, resources={r"/api/*": {"origins": "*"}})

CHROMA_DIR = Path("./chroma_db")
COLLECTION_NAME = "textbook_study"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class LocalChromaEmbeddings(Embeddings):
    """
    100% Free local ONNX embedding model (all-MiniLM-L6-v2).
    Runs offline on CPU with zero API cost and zero keys.
    """
    def __init__(self):
        self.ef = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        raw = self.ef(texts)
        return [[float(x) for x in vec] for vec in raw]

    def embed_query(self, text):
        raw = self.ef([text])[0]
        return [float(x) for x in raw]


_vector_store_cache = None
_vector_store_count = 0


def get_embeddings():
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and "your" not in openai_key and len(openai_key) > 15:
        return OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    return LocalChromaEmbeddings()


def get_vector_store():
    global _vector_store_cache, _vector_store_count
    if _vector_store_cache is not None:
        return _vector_store_cache, _vector_store_count

    if not CHROMA_DIR.exists():
        return None, 0

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collections = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME not in collections:
        return None, 0

    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    _vector_store_count = collection.count()
    if _vector_store_count == 0:
        return None, 0

    embeddings = get_embeddings()
    _vector_store_cache = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return _vector_store_cache, _vector_store_count


def get_llm(temperature=0.0):
    groq_key = os.getenv("GROQ_KEY") or os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key and "your" not in groq_key and len(groq_key) > 10:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=DEFAULT_GROQ_MODEL,
            groq_api_key=groq_key,
            temperature=temperature,
        ), "Groq", DEFAULT_GROQ_MODEL
    elif openai_key and "your" not in openai_key and len(openai_key) > 15:
        return ChatOpenAI(
            model=DEFAULT_OPENAI_MODEL,
            temperature=temperature,
        ), "OpenAI", DEFAULT_OPENAI_MODEL
    else:
        raise ValueError("No valid GROQ_KEY or OPENAI_API_KEY configured in .env")


def optimize_search_queries(llm, user_query: str, chat_history: list) -> list[str]:
    """
    Intelligent Query Reformulator:
    1. Resolves pronouns from conversation history (e.g. 'explain that simpler').
    2. Corrects spelling mistakes and typos (e.g. 'ecnomics' -> 'economics').
    3. Translates ordinal numbers to textbook headings (e.g. '7th principle' -> 'Principle 7').
    4. Produces 2 distinct queries (a heading/keyword query and a concept description query).
    """
    history_lines = []
    if chat_history:
        for msg in chat_history[-4:]:
            role = "Student" if isinstance(msg, HumanMessage) else "Assistant"
            history_lines.append(f"{role}: {msg.content}")

    history_block = ""
    if history_lines:
        history_block = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"

    prompt = (
        f"{history_block}"
        f"You are a search query optimizer for a college economics textbook assistant.\n"
        f"Student question: '{user_query}'\n\n"
        f"Generate 2 focused search queries for the textbook vector database:\n"
        f"1. Heading/keyword query: Fix any typos (e.g. 'ecnomics' -> 'economics') and translate ordinals to exact textbook heading format (e.g. '7th principle' -> 'Principle 7'). Do NOT append generic words like 'economics' or 'textbook'.\n"
        f"2. Concept description query: The principle name or core concept title if known (e.g. 'Governments can sometimes improve market outcomes'), or the standalone core question.\n\n"
        f"Output ONLY the 2 queries, one per line. Do not include numbers, bullets, or extra text."
    )

    try:
        response = llm.invoke(prompt)
        queries = []
        for line in response.content.strip().splitlines():
            cleaned = re.sub(r'^\s*[\d\.\-\*\)\>\#]+\s*', '', line).strip(' "\'')
            if cleaned:
                queries.append(cleaned)
        if not queries:
            queries = [user_query]
        return queries[:2]
    except Exception:
        return [user_query]


def multi_query_retrieve(vector_store, queries: list[str], top_k: int = 6):
    """
    Retrieves chunks across all generated queries, deduplicates them, and returns top results.
    """
    all_chunks = {}
    per_query_k = max(3, (top_k // len(queries)) + 1) if queries else top_k

    for q in queries:
        try:
            docs = vector_store.similarity_search_with_relevance_scores(q, k=per_query_k)
            for doc, score in docs:
                key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0), doc.page_content[:80])
                if key not in all_chunks or score > all_chunks[key][1]:
                    all_chunks[key] = (doc, score)
        except Exception:
            try:
                docs = vector_store.similarity_search(q, k=per_query_k)
                for doc in docs:
                    key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0), doc.page_content[:80])
                    if key not in all_chunks:
                        all_chunks[key] = (doc, 0.5)
            except Exception:
                continue

    sorted_docs = sorted(all_chunks.values(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:top_k]]


def build_qa_chain(llm):
    qa_system_prompt = (
        "You are an expert study assistant helping a student understand their textbook.\n"
        "Answer the question using ONLY the provided textbook context.\n"
        "If the topic or answer is not covered in the textbook context, respond strictly with: "
        "'This topic is not covered in your textbook.' Do not speculate, invent, or use outside knowledge.\n"
        "Whenever you state a fact or definition, cite the source and page number in parentheses (e.g. [Page 44]).\n"
        "Format your answer with clear markdown headings, bullet points, and key terms in bold where appropriate.\n\n"
        "Context from textbook:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return create_stuff_documents_chain(llm, qa_prompt)


# --- REST Endpoints ---

@app.route("/api/status", methods=["GET"])
def get_status():
    """Healthcheck and database metadata status."""
    try:
        vector_store, chunk_count = get_vector_store()
        provider = "None"
        model_name = "None"
        try:
            _, provider, model_name = get_llm()
            has_llm = True
        except Exception:
            has_llm = False

        data_dir = Path("./data")
        pdf_files = [f.name for f in data_dir.glob("*.pdf")] if data_dir.exists() else []

        return jsonify({
            "status": "ready" if (vector_store and chunk_count > 0 and has_llm) else "needs_setup",
            "database_ready": bool(vector_store and chunk_count > 0),
            "chunk_count": chunk_count,
            "collection_name": COLLECTION_NAME,
            "provider": provider,
            "model": model_name,
            "embedding_type": "Local ONNX (all-MiniLM-L6-v2, 100% Free)" if not os.getenv("OPENAI_API_KEY") else "OpenAI (text-embedding-3-small)",
            "textbooks": pdf_files,
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main RAG Chat Endpoint.
    Body:
    {
        "message": "What is the 7th principle of economics?",
        "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
        "top_k": 6,
        "temperature": 0.0
    }
    """
    start_time = time.time()
    data = request.get_json(force=True, silent=True) or {}
    user_query = data.get("message", "").strip()

    if not user_query:
        return jsonify({"error": "Query message cannot be empty"}), 400

    raw_history = data.get("history", [])
    top_k = int(data.get("top_k", 6))
    temperature = float(data.get("temperature", 0.0))

    vector_store, chunk_count = get_vector_store()
    if not vector_store or chunk_count == 0:
        return jsonify({
            "error": "No indexed textbooks found in vector database. Please run 'py ingest.py' first.",
            "status": "no_vectors"
        }), 400

    try:
        llm, provider, model_name = get_llm(temperature=temperature)
    except Exception as e:
        return jsonify({"error": str(e), "status": "no_llm_key"}), 500

    # Build LangChain chat history
    chat_history = []
    for msg in raw_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            chat_history.append(HumanMessage(content=content))
        elif role == "assistant":
            chat_history.append(AIMessage(content=content))

    # 1. Multi-query optimization
    search_queries = optimize_search_queries(llm, user_query, chat_history)

    # 2. Vector retrieval
    retrieved_docs = multi_query_retrieve(vector_store, search_queries, top_k=top_k)

    # 3. Generate answer
    qa_chain = build_qa_chain(llm)
    try:
        response = qa_chain.invoke({
            "context": retrieved_docs,
            "input": user_query,
            "chat_history": chat_history,
        })
        answer = response if isinstance(response, str) else str(response)
    except Exception as e:
        return jsonify({"error": f"LLM generation failed: {str(e)}"}), 500

    # 4. Format citations
    citations = []
    for idx, doc in enumerate(retrieved_docs, 1):
        citations.append({
            "id": idx,
            "source": doc.metadata.get("source", "Textbook"),
            "page": doc.metadata.get("page", 1),
            "text": doc.page_content.strip(),
            "preview": doc.page_content.strip()[:240] + ("..." if len(doc.page_content.strip()) > 240 else ""),
        })

    elapsed_ms = int((time.time() - start_time) * 1000)

    return jsonify({
        "answer": answer,
        "citations": citations,
        "search_queries": search_queries,
        "latency_ms": elapsed_ms,
        "model": model_name,
        "provider": provider,
    })


@app.route("/api/suggested", methods=["GET"])
def suggested_questions():
    """Returns curated starter prompts for student exploration."""
    return jsonify({
        "questions": [
            "What is the 7th principle of economics?",
            "Explain opportunity cost with a real-world example.",
            "What causes inflation according to the textbook?",
            "How do market economies allocate resources through prices?",
            "What is market failure and what causes it?",
            "Explain the difference between positive and normative statements."
        ]
    })


FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serves the compiled React frontend from frontend/dist."""
    if FRONTEND_DIST.exists():
        if path != "" and (FRONTEND_DIST / path).exists():
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({
        "message": "Study-RAG Flask API is active. For the React frontend, start the Vite dev server with 'cd frontend && npm run dev' or build it with 'npm run build'."
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f">> [Study-RAG] Flask Backend running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


