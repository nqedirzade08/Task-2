from ingest import load_documents
from chunking import chunk_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from embeddings import embed_chunks, embed_query, EMBEDDING_MODEL
from vectorstore import get_collection, add_chunks, similarity_search

DOCUMENTS_FOLDER = "documents"

TEST_QUERIES = [
    "ADNSU-da Kompüter Mühəndisliyi ixtisası üzrə 2025-ci il qəbul balı neçə idi?",
    "ADNSU-da xarici tələbələr üçün illik təhsil haqqı neçə manatdır?",
    "BMU nə vaxt yaradılıb və hansı ixtisasları var?",
]


def build_index():
    documents = load_documents(DOCUMENTS_FOLDER)
    print(f"Yüklənmiş sənəd sayı: {len(documents)}")
    for doc in documents:
        print(f"  - {doc['source']} ({len(doc['text'])} simvol)")

    if not documents:
        print("Heç bir sənəd tapılmadı, çıxılır.")
        return None

    print(f"\nChunking parametrləri: chunk_size={DEFAULT_CHUNK_SIZE}, chunk_overlap={DEFAULT_CHUNK_OVERLAP}")
    chunks = chunk_documents(documents)
    print(f"Ümumi chunk sayı: {len(chunks)}\n")

    print(f"Embedding modeli: {EMBEDDING_MODEL}")
    try:
        embedded_chunks = embed_chunks(chunks, input_type="passage")
    except RuntimeError as e:
        print(f"\nEmbedding generasiyası uğursuz oldu: {e}")
        return None

    collection = get_collection(reset=True)
    count = add_chunks(embedded_chunks, collection=collection)
    print(f"\nVektor bazasına {count} chunk yazıldı (Chroma, persist qovluğu: chroma_db/)")

    return collection


def run_test_queries(collection, top_k: int = 3):
    print("\n=== Oxşarlıq Axtarışı Testləri ===")

    for query in TEST_QUERIES:
        print(f"\nSual: {query}")
        try:
            query_vector = embed_query(query)
        except RuntimeError as e:
            print(f"  Embedding xətası: {e}")
            continue

        hits = similarity_search(query_vector, top_k=top_k, collection=collection)

        if not hits:
            print("  Heç bir nəticə tapılmadı.")
            continue

        for rank, hit in enumerate(hits, start=1):
            preview = hit["text"][:150].replace("\n", " ")
            print(f"  {rank}) [{hit['chunk_id']}] məsafə={hit['distance']:.4f}")
            print(f"     {preview}...")


def main():
    print("=== RAG Checkpoint 3 — Vektor Saxlama + Oxşarlıq Axtarışı ===\n")

    collection = build_index()
    if collection is None:
        return

    run_test_queries(collection)


if __name__ == "__main__":
    main()
