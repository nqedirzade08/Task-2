import json
import pickle
from ingest import load_documents
from chunking import chunk_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from embeddings import embed_chunks, EMBEDDING_MODEL

DOCUMENTS_FOLDER = "documents"
OUTPUT_PICKLE = "chunks_with_embeddings.pkl"
OUTPUT_PREVIEW_JSON = "chunks_with_embeddings_preview.json"


def main():
    print("=== RAG Checkpoint 2 — Chunk-lar üçün Embedding Generasiyası ===\n")

    documents = load_documents(DOCUMENTS_FOLDER)
    print(f"Yüklənmiş sənəd sayı: {len(documents)}")
    for doc in documents:
        print(f"  - {doc['source']} ({len(doc['text'])} simvol)")

    if not documents:
        print("Heç bir sənəd tapılmadı, çıxılır.")
        return

    print(f"\nChunking parametrləri: chunk_size={DEFAULT_CHUNK_SIZE}, chunk_overlap={DEFAULT_CHUNK_OVERLAP}")
    chunks = chunk_documents(documents)
    print(f"Ümumi chunk sayı: {len(chunks)}\n")

    print(f"Embedding modeli: {EMBEDDING_MODEL}\n")
    try:
        embedded_chunks = embed_chunks(chunks, input_type="passage")
    except RuntimeError as e:
        print(f"\nEmbedding generasiyası uğursuz oldu: {e}")
        return

    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump(embedded_chunks, f)
    print(f"\nTam nəticə (vektorlarla birlikdə) '{OUTPUT_PICKLE}' faylına yazıldı.")

    preview = [
        {
            "chunk_id": c["chunk_id"],
            "source": c["source"],
            "text_preview": c["text"][:120] + ("..." if len(c["text"]) > 120 else ""),
            "embedding_dim": c["embedding_dim"],
            "embedding_preview": c["embedding"][:5],
        }
        for c in embedded_chunks
    ]
    with open(OUTPUT_PREVIEW_JSON, "w", encoding="utf-8") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)
    print(f"Oxunaqlı önizləmə '{OUTPUT_PREVIEW_JSON}' faylına yazıldı (vektorun ilk 5 ölçüsü ilə).")

    if embedded_chunks:
        dim = embedded_chunks[0]["embedding_dim"]
        print(f"\nHər chunk üçün vektor ölçüsü: {dim}")
        print(f"Cəmi embedding edilmiş chunk: {len(embedded_chunks)}")


if __name__ == "__main__":
    main()
