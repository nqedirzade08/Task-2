import json
from ingest import load_documents
from chunking import chunk_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

DOCUMENTS_FOLDER = "documents"
OUTPUT_FILE = "chunks_output.json"


def main():
    print("=== RAG Checkpoint 1 — Sənəd Ingestion + Chunking ===\n")

    documents = load_documents(DOCUMENTS_FOLDER)
    print(f"Yüklənmiş sənəd sayı: {len(documents)}")
    for doc in documents:
        print(f"  - {doc['source']} ({len(doc['text'])} simvol)")

    if not documents:
        print("Heç bir sənəd tapılmadı, çıxılır.")
        return

    print(f"\nChunking parametrləri: chunk_size={DEFAULT_CHUNK_SIZE}, chunk_overlap={DEFAULT_CHUNK_OVERLAP}\n")

    chunks = chunk_documents(documents)
    print(f"Ümumi chunk sayı: {len(chunks)}\n")

    per_source = {}
    for c in chunks:
        per_source.setdefault(c["source"], 0)
        per_source[c["source"]] += 1

    for source, count in per_source.items():
        print(f"  - {source}: {count} chunk")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nChunk-lar '{OUTPUT_FILE}' faylına yazıldı.")


if __name__ == "__main__":
    main()
