import os

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_DISTANCE_THRESHOLD", "0.50"))

CITATION_SYSTEM_INSTRUCTIONS = """Sən yalnız istifadəçiyə verilən KONTEKST bölməsindəki məlumata əsaslanaraq cavab verən sənəd-əsaslı köməkçisən.

QAYDALAR:
1. Yalnız KONTEKST bölməsində olan məlumatdan istifadə et. Öz ümumi biliyindən, təxminlərdən və ya uydurmadan istifadə ETMƏ.
2. Əgər KONTEKST sualın cavabını ehtiva etmirsə, "answer" sahəsində bunu aydın bildir: "Bu suala cavab verə bilmirəm, çünki mövcud sənədlərdə bu barədə məlumat yoxdur." Bu halda "sources" siyahısını boş burax.
3. [SİSTEM QEYDİ] işarəli xəbərdarlıq gəlsə, tapılan məlumatın sualla zəif əlaqəli ola biləcəyini göstərir — xüsusilə ehtiyatlı ol, uydurma cavab vermə.
4. Hər KONTEKST parçası [C1], [C2], [C3] kimi etiketlənib. Cavabında İSTİFADƏ ETDİYİN parçaların etiketlərini "sources" siyahısında qeyd et. Yalnız aşağıda verilmiş etiketlərdən istifadə et, yeni etiket uydurma.
5. Cavabın YALNIZ aşağıdakı JSON formatında olmalıdır, başqa heç bir mətn, izahat və ya markdown kod bloku əlavə etmə:

{"answer": "cavabın mətni", "sources": ["C1", "C2"]}
"""


def format_labeled_context(hits: list) -> tuple:
    parts = []
    label_map = {}
    for i, hit in enumerate(hits, start=1):
        label = f"C{i}"
        label_map[label] = hit
        meta = hit["metadata"]
        parts.append(
            f"[{label}] Mənbə: {meta['source']} (chunk {meta['chunk_index']}, "
            f"oxşarlıq məsafəsi: {hit['distance']:.4f})\n{hit['text']}"
        )
    return "\n\n".join(parts), label_map


def build_citation_messages(query: str, hits: list) -> tuple:
    if not hits:
        context_block = "(Heç bir uyğun chunk tapılmadı.)"
        label_map = {}
        best_distance = None
    else:
        context_block, label_map = format_labeled_context(hits)
        best_distance = hits[0]["distance"]

    low_confidence = best_distance is None or best_distance > SIMILARITY_THRESHOLD

    confidence_note = ""
    if low_confidence:
        distance_str = f"{best_distance:.4f}" if best_distance is not None else "N/A"
        confidence_note = (
            f"\n\n[SİSTEM QEYDİ: Ən yaxın tapılan parçanın oxşarlıq məsafəsi ({distance_str}) "
            f"{SIMILARITY_THRESHOLD} həddini aşır — bu, sualın KONTEKST-də birbaşa cavablandırılmaya "
            "bilməyəcəyini göstərir. Bu halda uydurma cavab vermə, məlumatın olmadığını bildir.]"
        )

    user_content = (
        "[KONTEKST]\n"
        f"{context_block}\n"
        "[/KONTEKST]"
        f"{confidence_note}\n\n"
        "[SUAL]\n"
        f"{query}\n"
        "[/SUAL]"
    )

    messages = [
        {"role": "system", "content": CITATION_SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_content},
    ]

    return messages, label_map, low_confidence
