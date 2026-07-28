from ingest import load_documents
from chunking import chunk_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from embeddings import embed_chunks, embed_query, EMBEDDING_MODEL
from vectorstore import get_collection, add_chunks, similarity_search
from prompt_builder import build_rag_messages, SIMILARITY_THRESHOLD
from generation import generate_answer, CHAT_MODEL

DOCUMENTS_FOLDER = "documents"
TOP_K = 3

TEST_QUERIES = [
    "ADNSU-da Kompüter Mühəndisliyi ixtisası üzrə 2025-ci il qəbul balı neçə idi?",
    "ADNSU-da xarici tələbələr üçün illik təhsil haqqı neçə manatdır?",
]


def build_index():
    documents = load_documents(DOCUMENTS_FOLDER)
    print(f"Yüklənmiş sənəd sayı: {len(documents)}")

    if not documents:
        print("Heç bir sənəd tapılmadı, çıxılır.")
        return None

    chunks = chunk_documents(documents)
    print(f"Ümumi chunk sayı: {len(chunks)}")

    print(f"Embedding modeli: {EMBEDDING_MODEL}")
    try:
        embedded_chunks = embed_chunks(chunks, input_type="passage")
    except RuntimeError as e:
        print(f"Embedding generasiyası uğursuz oldu: {e}")
        return None

    collection = get_collection(reset=True)
    add_chunks(embedded_chunks, collection=collection)
    print("Vektor bazası hazırdır.\n")

    return collection


def answer_query(query: str, collection):
    print("=" * 70)
    print(f"SUAL: {query}\n")

    try:
        query_vector = embed_query(query)
    except RuntimeError as e:
        print(f"Embedding xətası: {e}")
        return

    hits = similarity_search(query_vector, top_k=TOP_K, collection=collection)

    messages, low_confidence = build_rag_messages(query, hits)

    print("--- QURULMUŞ PROMPT ---\n")
    print("[SYSTEM]")
    print(messages[0]["content"])
    print("\n[USER]")
    print(messages[1]["content"])
    print("\n--- PROMPT SONU ---\n")

    if low_confidence:
        print(f"(Aşağı etibarlılıq siqnalı aktivdir — ən yaxın məsafə həddi {SIMILARITY_THRESHOLD}-dən yüksəkdir)\n")

    print(f"Model: {CHAT_MODEL}")
    print("Cavab generasiya olunur...\n")
    answer = generate_answer(messages)

    print("--- MODEL CAVABI ---")
    print(answer)
    print()


def main():
    print("=== RAG Checkpoint 4 — Retrieval + Prompt Qurulması ===\n")

    collection = build_index()
    if collection is None:
        return

    for query in TEST_QUERIES:
        answer_query(query, collection)


if __name__ == "__main__":
    main()
