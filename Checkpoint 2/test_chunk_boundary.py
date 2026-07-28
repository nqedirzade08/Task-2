from ingest import load_documents
from chunking import naive_fixed_chunk, sliding_window_chunk, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

KEY_FACT = (
    "Kompüter Mühəndisliyi ixtisası üzrə 2025-ci il qəbul imtahanında ən aşağı keçid balı "
    "342.7 bal olub, bu da əvvəlki ilə nisbətən 15 bal yüksəkdir."
)


def run_boundary_test():
    print("=== Chunk Sərhədi Testi (overlap-un vacibliyi) ===\n")

    documents = load_documents("documents")
    doc = next(d for d in documents if d["source"] == "adnsu_melumati.txt")
    text = doc["text"]

    fact_start = text.find(KEY_FACT)
    fact_end = fact_start + len(KEY_FACT)
    print(f"Açar fakt sənəddə {fact_start}-{fact_end} simvol aralığındadır.\n")

    naive_chunks = naive_fixed_chunk(text, DEFAULT_CHUNK_SIZE)
    naive_intact = any(KEY_FACT in c for c in naive_chunks)

    print(f"1) NAIVE fixed-size chunking (chunk_size={DEFAULT_CHUNK_SIZE}, overlap=0)")
    print(f"   Chunk sayı: {len(naive_chunks)}")
    print(f"   Açar fakt TAM bir chunk-dadır: {naive_intact}")
    if not naive_intact:
        boundary_chunk_idx = fact_start // DEFAULT_CHUNK_SIZE
        cut_point = DEFAULT_CHUNK_SIZE - (fact_start % DEFAULT_CHUNK_SIZE)
        print(f"   Kəsilmə nöqtəsi faktın {cut_point}-ci simvolundadır:")
        print(f"     chunk {boundary_chunk_idx} sonu:   ...{naive_chunks[boundary_chunk_idx][-40:]!r}")
        print(f"     chunk {boundary_chunk_idx + 1} əvvəli: {naive_chunks[boundary_chunk_idx + 1][:40]!r}...")

    spans = sliding_window_chunk(text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    overlap_chunks = [text[s:e] for s, e in spans]
    overlap_intact = any(KEY_FACT in c for c in overlap_chunks)

    print(f"\n2) OVERLAP chunking (chunk_size={DEFAULT_CHUNK_SIZE}, overlap={DEFAULT_CHUNK_OVERLAP})")
    print(f"   Chunk sayı: {len(overlap_chunks)}")
    print(f"   Açar fakt TAM bir chunk-dadır: {overlap_intact}")
    for i, c in enumerate(overlap_chunks):
        if KEY_FACT in c:
            print(f"   -> chunk {i}-də tam şəkildə mövcuddur (chunk uzunluğu: {len(c)})")

    print("\n=== Nəticə ===")
    if not naive_intact and overlap_intact:
        print("TƏSDİQLƏNDİ: overlap olmadan açar fakt iki chunk arasında bölünür,")
        print("overlap strategiyası ilə isə fakt heç olmasa bir chunk-da tam saxlanılır.")
    else:
        print("Gözlənilməz nəticə — parametrləri yenidən yoxlayın.")


if __name__ == "__main__":
    run_boundary_test()
