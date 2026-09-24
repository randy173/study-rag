import sys
sys.stdout.reconfigure(encoding='utf-8')

import chromadb
import chromadb.utils.embedding_functions as ef

client = chromadb.PersistentClient(path="chroma_db")
col = client.get_collection("textbook_study")
fn = ef.DefaultEmbeddingFunction()

queries = [
    "What is the 7th principle of ecnomics",       # exact user prompt with typo
    "What is the 7th principle of economics",      # user prompt without typo
    "What is the seventh principle of economics?", # written out ordinal
    "Principle 7",                                 # keyword
    "Governments Can Sometimes Improve Market Outcomes" # target text
]

for q in queries:
    print("\n" + "=" * 70)
    print(f"QUERY: '{q}'")
    print("=" * 70)
    q_emb = [[float(x) for x in vec] for vec in fn([q])]
    res = col.query(query_embeddings=q_emb, n_results=5)
    for i in range(len(res["ids"][0])):
        page = res["metadatas"][0][i]["page"]
        dist = res["distances"][0][i]
        doc = res["documents"][0][i].replace("\n", " ")[:120]
        print(f"Rank {i+1} | Page {page} | Dist {dist:.4f} | {doc}...")
