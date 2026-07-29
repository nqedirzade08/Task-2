from ingest import load_documents
from chunking import chunk_documents
from embeddings import embed_chunks, embed_query, EMBEDDING_MODEL
from vectorstore import get_collection, add_chunks, similarity_search
from prompt_builder import build_citation_messages, SIMILARITY_THRESHOLD
from generation import generate_citation_answer, CHAT_MODEL
from citation import format_answer_with_citations

DOCUMENTS_FOLDER = "documents"
TOP_K = 3

TEST_CASES = [
    ("ADNSU-da Kompüter Mühəndisliyi ixtisası üzrə 2025-ci il qəbul balı neçə idi?", True),
    ("BMU nə vaxt yaradılıb?", True),
    ("ADNSU-nun kitabxana fondunda neçə kitab var?", True),
    ("ADNSU-da xarici tələbələr üçün illik təhsil haqqı neçə manatdır?", False),
    ("BMU-nun rektoru kimdir?", False),
    ("ADNSU-nun kampusunda neçə tələbə yataqxanası var?", False),
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


def run_case(query: str, expected_found: bool, collection) -> bool:
    print("=" * 70)
    print(f"SUAL: {query}")
    print(f"Gözlənilən: {'CAVAB VAR' if expected_found else 'CAVAB YOXDUR'}")

    try:
        query_vector = embed_query(query)
    except RuntimeError as e:
        print(f"Embedding xətası: {e}")
        return False

    hits = similarity_search(query_vector, top_k=TOP_K, collection=collection)
    messages, label_map, low_confidence = build_citation_messages(query, hits)

    result = generate_citation_answer(messages, label_map)

    if "error" in result:
        print(f"[Xəta] {result['error']}")
        return False

    actual_found = result["answer_found_in_context"]
    passed = actual_found == expected_found

    print(f"Model qərarı: {'CAVAB VAR' if actual_found else 'CAVAB YOXDUR'} — {'DOĞRU' if passed else 'SƏHV'}")
    print(format_answer_with_citations(result))
    print()

    return passed


def main():
    print("=== RAG Checkpoint 6 — 'Sənədlərdə Yoxdur' Halının İdarə Olunması ===\n")
    print(f"Model: {CHAT_MODEL}, oxşarlıq həddi: {SIMILARITY_THRESHOLD}\n")

    collection = build_index()
    if collection is None:
        return

    results = []
    for query, expected in TEST_CASES:
        passed = run_case(query, expected, collection)
        results.append((query, expected, passed))

    print("=" * 70)
    print("=== YEKUN NƏTİCƏ ===\n")
    correct = sum(1 for _, _, p in results if p)
    total = len(results)

    for query, expected, passed in results:
        status = "✓" if passed else "✗"
        label = "CAVAB VAR" if expected else "CAVAB YOXDUR"
        print(f"  [{status}] ({label}) {query}")

    print(f"\nDoğruluq: {correct}/{total} ({100 * correct / total:.0f}%)")


if __name__ == "__main__":
    main()
