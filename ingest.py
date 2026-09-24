import os
import sys
import glob
import time
import hashlib
import re
from pathlib import Path
from dotenv import load_dotenv

# PyMuPDF
import pymupdf

# LangChain components
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings

load_dotenv()

DATA_DIR = Path("./data")
CHROMA_DIR = Path("./chroma_db")
COLLECTION_NAME = "textbook_study"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
MAX_RETRIES = 5


def clean_text(text: str) -> str:
    """
    Normalizes textbook text:
    - Removes hyphenated line breaks (e.g. 'din-\\nner' -> 'dinner')
    - Fixes font replacement characters for apostrophes and quotation marks
    - Cleans irregular spacing
    """
    if not text:
        return ""
    # Reconnect words hyphenated across lines (e.g. din-\nner -> dinner)
    text = re.sub(r'(\b\w+)-\n(\w+\b)', r'\1\2', text)
    # Normalize unicode curly quotes and dashes to standard ASCII
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '--')
    # Replace remaining unprintable/replacement characters with quotes
    text = text.replace("\ufffd", '"')
    # Collapse multiple inline spaces
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


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
    """
    Detects available provider:
    - If OPENAI_API_KEY is present, uses OpenAIEmbeddings.
    - Otherwise (e.g. Groq user), uses fast local ONNX embeddings.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and "your" not in openai_key and len(openai_key) > 15:
        print("[Embedder] Using OpenAI 'text-embedding-3-small'")
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)
    else:
        print("[Embedder] Using local ONNX embeddings (all-MiniLM-L6-v2) — 100% free & local")
        return LocalChromaEmbeddings()


def check_api_key():
    groq_key = os.getenv("GROQ_KEY") or os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    has_groq = groq_key and "your" not in groq_key and len(groq_key) > 10
    has_openai = openai_key and "your" not in openai_key and len(openai_key) > 10
    
    if not has_groq and not has_openai:
        print("[ERROR] Neither GROQ_KEY nor OPENAI_API_KEY found in your .env file.")
        print("Please edit .env and insert your Groq or OpenAI API key.")
        sys.exit(1)
        
    if has_groq and not has_openai:
        print("[Config] Detected GROQ_KEY. Using local ONNX embeddings for vectors + Groq for LLM!")


def is_header_or_footer(bbox, page_rect):
    """
    Detect if a block falls into the top or bottom 5% of the page
    or within 40pt margin (common for running headers and footers).
    """
    top_margin = max(40, page_rect.height * 0.05)
    bottom_margin = page_rect.height - max(40, page_rect.height * 0.05)
    
    # bbox format: (x0, y0, x1, y1)
    y0, y1 = bbox[1], bbox[3]
    if y1 <= top_margin or y0 >= bottom_margin:
        return True
    return False


def extract_pdf_pages(pdf_path: Path):
    """
    Extracts text from a PDF file page by page using PyMuPDF blocks.
    Uses block sorting to respect column order and filters out header/footer noise.
    """
    documents = []
    print(f"\n[Ingest] Processing '{pdf_path.name}'...")
    
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    print(f" -> Found {total_pages} pages.")
    
    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_rect = page.rect
        # sort=True sorts blocks in standard reading order
        blocks = page.get_text("blocks", sort=True)
        
        page_text_pieces = []
        for b in blocks:
            # b: (x0, y0, x1, y1, text, block_no, block_type)
            # block_type 0 is text, 1 is image
            if len(b) >= 7 and b[6] != 0:
                continue
            
            block_bbox = (b[0], b[1], b[2], b[3])
            block_text = b[4].strip()
            
            if not block_text:
                continue
            
            # Check for header/footer noise or standalone page numbers
            if is_header_or_footer(block_bbox, page_rect):
                if block_text.isdigit() or len(block_text.split()) <= 6:
                    continue  # Strip running header/footer
            
            page_text_pieces.append(block_text)
            
        combined_text = clean_text("\n\n".join(page_text_pieces))
        if combined_text:
            documents.append(
                Document(
                    page_content=combined_text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_idx + 1,  # 1-indexed for human readability
                    }
                )
            )
            
    doc.close()
    print(f" -> Extracted content from {len(documents)} non-empty pages.")
    return documents


def chunk_text_with_tolerance(text: str, target_size: int = 800, tolerance: int = 150, overlap: int = 150, min_chunk_words: int = 15):
    """
    Adaptive Chunking with Tolerance:
    - Splits text on natural paragraph (\\n\\n) and sentence boundaries.
    - Uses a soft target window [target_size - tolerance, target_size + tolerance].
    - Sentence-snaps overlap so chunks never start or end mid-sentence.
    - Absorbs small leftover remnants into the previous chunk instead of creating orphan tails.
    - Discards non-semantic noise chunks under min_chunk_words.
    """
    raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    elements = []
    for p in raw_paragraphs:
        sents = re.split(r'(?<=[.?!])\s+', p)
        for s in sents:
            s_clean = s.strip()
            if s_clean:
                elements.append(s_clean)

    if not elements:
        return []

    chunks = []
    current_elements = []
    current_len = 0

    for el in elements:
        el_len = len(el)
        # If adding this sentence fits within target + tolerance, keep accumulating
        if current_len + el_len <= (target_size + tolerance):
            current_elements.append(el)
            current_len += el_len + 1
        else:
            # Commit current chunk
            if current_elements:
                chunks.append(' '.join(current_elements))

            # Build sentence-snapped overlap buffer from the tail of current chunk
            overlap_elements = []
            overlap_len = 0
            for prev in reversed(current_elements):
                if overlap_len + len(prev) <= overlap:
                    overlap_elements.insert(0, prev)
                    overlap_len += len(prev) + 1
                else:
                    break

            current_elements = overlap_elements + [el]
            current_len = sum(len(x) for x in current_elements) + len(current_elements)

    # Handle final remnant: absorb into last chunk if small, otherwise append
    if current_elements:
        final_text = ' '.join(current_elements)
        if chunks and len(final_text) <= (target_size // 2):
            new_elements = [el for el in current_elements if el not in chunks[-1]]
            if new_elements:
                chunks[-1] += ' ' + ' '.join(new_elements)
        else:
            chunks.append(final_text)

    # Discard non-semantic noise fragments (e.g. photo credits, single words)
    return [c.strip() for c in chunks if len(c.strip().split()) >= min_chunk_words]


def chunk_documents(documents, target_size: int = 800, tolerance: int = 150, overlap: int = 150):
    """
    Applies adaptive tolerance chunking across all extracted documents.
    """
    all_chunks = []
    for doc in documents:
        text_chunks = chunk_text_with_tolerance(
            doc.page_content,
            target_size=target_size,
            tolerance=tolerance,
            overlap=overlap
        )
        for chunk_idx, text in enumerate(text_chunks):
            chunk_meta = dict(doc.metadata)
            chunk_meta["chunk_idx"] = chunk_idx
            all_chunks.append(Document(page_content=text, metadata=chunk_meta))

    print(f"[Chunker] Generated {len(all_chunks)} adaptive chunks with tolerance (no orphan fragments).")
    return all_chunks


def generate_chunk_id(source: str, page: int, text: str) -> str:
    """
    Generates a deterministic MD5 hash for idempotency and deduplication.
    """
    raw_key = f"{source}_{page}_{text}".strip()
    return hashlib.md5(raw_key.encode("utf-8")).hexdigest()


def embed_and_upsert(chunks):
    """
    Embeds chunks in batches with exponential backoff and upserts into ChromaDB.
    Skips chunks that are already embedded to save API cost and time.
    """
    if not chunks:
        print("[Warning] No chunks to embed.")
        return

    # Initialize ChromaDB persistent client
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if "--reset" in sys.argv:
        print(f"[VectorDB] '--reset' flag detected. Wiping collection '{COLLECTION_NAME}' for clean re-ingest...")
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    # Initialize Embeddings (OpenAI or local ONNX)
    embeddings = get_embeddings()

    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Build records with deterministic IDs
    all_ids = []
    all_texts = []
    all_metadatas = []
    
    for chunk in chunks:
        cid = generate_chunk_id(
            chunk.metadata.get("source", "unknown"),
            chunk.metadata.get("page", 0),
            chunk.page_content
        )
        all_ids.append(cid)
        all_texts.append(chunk.page_content)
        all_metadatas.append(chunk.metadata)

    # Check for existing IDs to skip duplicate embedding costs
    print(f"[VectorDB] Checking existing chunks in Chroma collection '{COLLECTION_NAME}'...")
    existing_records = collection.get(ids=all_ids)
    existing_ids = set(existing_records["ids"]) if existing_records and "ids" in existing_records else set()
    
    new_ids = []
    new_texts = []
    new_metadatas = []
    
    for cid, txt, meta in zip(all_ids, all_texts, all_metadatas):
        if cid not in existing_ids:
            new_ids.append(cid)
            new_texts.append(txt)
            new_metadatas.append(meta)
            
    skipped_count = len(all_ids) - len(new_ids)
    if skipped_count > 0:
        print(f" -> Found {skipped_count} existing chunks. Skipping them to save API cost.")
    
    if not new_ids:
        print("[VectorDB] All chunks are already up to date in ChromaDB! Nothing new to embed.")
        print(f"Total chunks in collection: {collection.count()}")
        return

    print(f"[Embedder] Embedding and upserting {len(new_ids)} new chunks in batches of {BATCH_SIZE}...")
    
    for i in range(0, len(new_ids), BATCH_SIZE):
        batch_ids = new_ids[i:i + BATCH_SIZE]
        batch_texts = new_texts[i:i + BATCH_SIZE]
        batch_metadatas = new_metadatas[i:i + BATCH_SIZE]
        
        # Exponential backoff retry loop
        retries = 0
        success = False
        while retries < MAX_RETRIES and not success:
            try:
                batch_embeddings = embeddings.embed_documents(batch_texts)
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )
                success = True
                processed = min(i + BATCH_SIZE, len(new_ids))
                pct = int((processed / len(new_ids)) * 100)
                print(f" -> Progress: {processed}/{len(new_ids)} chunks ({pct}%) upserted.")
            except Exception as e:
                retries += 1
                wait_time = 2 ** retries
                print(f"[Warning] Error embedding batch (attempt {retries}/{MAX_RETRIES}): {e}")
                if retries >= MAX_RETRIES:
                    print(f"[Error] Failed to embed batch after {MAX_RETRIES} attempts. Aborting.")
                    raise
                print(f" -> Retrying in {wait_time}s...")
                time.sleep(wait_time)

    print(f"\n[Success] Ingestion complete! Total chunks in collection: {collection.count()}")


def main():
    print("=" * 60)
    print("  STUDY-RAG OFFLINE INGESTION PIPELINE")
    print("=" * 60)
    
    check_api_key()
    
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[ERROR] No PDF files found in '{DATA_DIR.resolve()}'.")
        print("Please place your textbook PDF(s) into the 'data/' folder and rerun.")
        sys.exit(1)
        
    print(f"Found {len(pdf_files)} PDF(s) in {DATA_DIR}:")
    for p in pdf_files:
        print(f" - {p.name} ({round(p.stat().st_size / (1024 * 1024), 2)} MB)")
        
    all_documents = []
    for pdf_path in pdf_files:
        docs = extract_pdf_pages(pdf_path)
        all_documents.extend(docs)
        
    chunks = chunk_documents(all_documents)
    embed_and_upsert(chunks)


if __name__ == "__main__":
    main()
