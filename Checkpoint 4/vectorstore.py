import chromadb

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "sened_chunklari"


def get_client():
    return chromadb.PersistentClient(path=PERSIST_DIR)


def get_collection(client=None, reset=False):
    client = client or get_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(embedded_chunks: list, collection=None) -> int:
    collection = collection or get_collection()

    ids = [c["chunk_id"] for c in embedded_chunks]
    embeddings = [c["embedding"] for c in embedded_chunks]
    documents = [c["text"] for c in embedded_chunks]
    metadatas = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "start_char": c["start_char"],
            "end_char": c["end_char"],
        }
        for c in embedded_chunks
    ]

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(ids)


def similarity_search(query_embedding: list, top_k: int = 3, collection=None) -> list:
    collection = collection or get_collection()

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    hits = []
    if not results["ids"] or not results["ids"][0]:
        return hits

    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return hits
