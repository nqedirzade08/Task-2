import os

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_DISTANCE_THRESHOLD", "0.50"))

SYSTEM_INSTRUCTIONS = """Sən yalnız istifadəçiyə verilən KONTEKST bölməsindəki məlumata əsaslanaraq cavab verən sənəd-əsaslı köməkçisən.

QAYDALAR:
1. Yalnız KONTEKST bölməsində olan məlumatdan istifadə et. Öz ümumi biliyindən, təxminlərdən və ya uydurmadan istifadə ETMƏ.
2. Əgər KONTEKST sualın cavabını ehtiva etmirsə, aydın şəkildə bunu bildir: "Bu suala cavab verə bilmirəm, çünki mövcud sənədlərdə bu barədə məlumat yoxdur." Heç bir rəqəm, tarix və ya fakt UYDURMA.
3. [SİSTEM QEYDİ] işarəli xəbərdarlıq gəlsə, bu, tapılan məlumatın sualla zəif əlaqəli ola biləcəyini göstərir — bu halda xüsusilə ehtiyatlı ol və əmin olmadığın halda uydurma cavab vermə.
4. Cavabında mümkün olduqda hansı mənbədən (sənəd adı) istifadə etdiyini qeyd et.
5. Cavabı qısa, dəqiq və birbaşa sualı hədəf alan formada ver.
"""


def format_context_block(hits: list) -> str:
    parts = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        parts.append(
            f"--- Mənbə {i}: {meta['source']} (chunk {meta['chunk_index']}, "
            f"oxşarlıq məsafəsi: {hit['distance']:.4f}) ---\n{hit['text']}"
        )
    return "\n\n".join(parts)


def build_rag_messages(query: str, hits: list) -> tuple:
    if not hits:
        context_block = "(Heç bir uyğun chunk tapılmadı.)"
        best_distance = None
    else:
        context_block = format_context_block(hits)
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
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_content},
    ]

    return messages, low_confidence
