import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# LangChain & ChromaDB
import chromadb
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Page configuration
st.set_page_config(
    page_title="Study-RAG | Textbook Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

CHROMA_DIR = Path("./chroma_db")
COLLECTION_NAME = "textbook_study"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# Custom CSS for polished look
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #888888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .citation-box {
        background-color: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #4CAF50;
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


from langchain_core.embeddings import Embeddings
import chromadb.utils.embedding_functions as ef

class LocalChromaEmbeddings(Embeddings):
    """
    100% Free local ONNX embedding model (all-MiniLM-L6-v2).
    Runs offline on your CPU with zero API costs and zero keys needed.
    """
    def __init__(self):
        self.ef = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        raw = self.ef(texts)
        return [[float(x) for x in vec] for vec in raw]

    def embed_query(self, text):
        raw = self.ef([text])[0]
        return [float(x) for x in raw]


def get_embeddings():
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and "your" not in openai_key and len(openai_key) > 15:
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return LocalChromaEmbeddings()


def get_llm(temperature=0.0):
    groq_key = os.getenv("GROQ_KEY") or os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if groq_key and "your" not in groq_key and len(groq_key) > 10:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=groq_key,
            temperature=temperature,
        )
    elif openai_key and "your" not in openai_key and len(openai_key) > 15:
        return ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            temperature=temperature,
        )
    else:
        st.error("No valid API key found for Groq or OpenAI in .env.")
        st.stop()


def verify_setup():
    groq_key = os.getenv("GROQ_KEY") or os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    has_groq = groq_key and "your" not in groq_key and len(groq_key) > 10
    has_openai = openai_key and "your" not in openai_key and len(openai_key) > 15
    
    if not has_groq and not has_openai:
        st.error("⚠️ **No API Key Configured**")
        st.info("Please open your `.env` file and add your `GROQ_KEY` or `OPENAI_API_KEY`, then reload.")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_vector_store():
    """
    Initializes and caches the ChromaDB client and vector store.
    """
    if not CHROMA_DIR.exists():
        return None, 0

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Check if collection exists
    collections = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME not in collections:
        return None, 0

    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    chunk_count = collection.count()
    if chunk_count == 0:
        return None, 0

    embeddings = get_embeddings()
    vector_store = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return vector_store, chunk_count


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
        queries = [
            line.strip().strip('- 1234567890."\'')
            for line in response.content.strip().splitlines()
            if line.strip()
        ]
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
            # Fallback to standard similarity_search if relevance scores not supported
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
    """
    Constructs the strict, citation-grounded answering chain.
    """
    qa_system_prompt = (
        "You are an expert study assistant helping a student understand their textbook.\n"
        "Answer the question using ONLY the provided textbook context.\n"
        "If the topic or answer is not covered in the textbook context, respond strictly with: "
        "'This topic is not covered in your textbook.' Do not speculate, invent, or use outside knowledge.\n"
        "Whenever you state a fact or definition, cite the source and page number in parentheses (e.g. [Page 44]).\n\n"
        "Context from textbook:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return create_stuff_documents_chain(llm, qa_prompt)


def main():
    verify_setup()

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings & Status")
        vector_store, chunk_count = load_vector_store()
        
        if vector_store and chunk_count > 0:
            st.success(f"✅ **Database Ready**: {chunk_count:,} chunks indexed")
        else:
            st.warning("⚠️ **No indexed textbooks found.**")

        top_k = st.slider("Retrieved chunks (k)", min_value=2, max_value=12, value=6, step=1)
        temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        
        st.markdown("---")
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("Architecture: PyMuPDF -> RecursiveSplitter -> ChromaDB -> LCEL Multi-Query RAG")

    # Main area
    st.markdown('<div class="main-title">📚 Study-RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Ask questions, explore definitions, and clarify concepts directly from your textbook.</div>',
        unsafe_allow_html=True,
    )

    if not vector_store or chunk_count == 0:
        st.warning(
            """
            ### No vectors found in `chroma_db`!
            To start asking questions:
            1. Place your textbook PDF(s) into the `data/` folder.
            2. Run the offline ingestion command in your terminal:
               ```bash
               py ingest.py
               ```
            3. Once ingestion is complete, refresh this page.
            """
        )
        return

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior conversation messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📖 Sources ({len(msg['citations'])} excerpts)"):
                    if msg.get("search_queries"):
                        st.caption(f"🔍 **Optimized Search**: `{', '.join(msg['search_queries'])}`")
                    for idx, cit in enumerate(msg["citations"], 1):
                        st.markdown(
                            f"**[{idx}] {cit['source']} — Page {cit['page']}**\n\n"
                            f"> {cit['text']}"
                        )

    # Chat input
    if user_query := st.chat_input("Ask a question about your textbook..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Build conversational history
        chat_history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        # Execute query
        with st.chat_message("assistant"):
            with st.spinner("Analyzing question & searching textbook..."):
                try:
                    llm = get_llm(temperature=temperature)
                    qa_chain = build_qa_chain(llm)

                    # 1. Optimize queries (fix typos, expand ordinals, resolve context)
                    search_queries = optimize_search_queries(llm, user_query, chat_history)

                    # 2. Multi-query retrieval
                    retrieved_docs = multi_query_retrieve(vector_store, search_queries, top_k=top_k)

                    # 3. Generate answer
                    response = qa_chain.invoke({
                        "context": retrieved_docs,
                        "input": user_query,
                        "chat_history": chat_history,
                    })
                    answer = response if isinstance(response, str) else str(response)

                    st.markdown(answer)

                    # Extract citations
                    citations = []
                    for doc in retrieved_docs:
                        citations.append({
                            "source": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page", "N/A"),
                            "text": doc.page_content.strip()[:300] + "...",
                        })

                    if citations:
                        with st.expander(f"📖 Sources ({len(citations)} excerpts)"):
                            st.caption(f"🔍 **Optimized Search**: `{', '.join(search_queries)}`")
                            for idx, cit in enumerate(citations, 1):
                                st.markdown(
                                    f"**[{idx}] {cit['source']} — Page {cit['page']}**\n\n"
                                    f"> {cit['text']}"
                                )

                    # Save to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "search_queries": search_queries,
                    })

                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")


if __name__ == "__main__":
    main()
