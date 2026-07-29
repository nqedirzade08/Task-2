DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def naive_fixed_chunk(text: str, chunk_size: int) -> list:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def sliding_window_chunk(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                          chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap chunk_size-dən kiçik olmalıdır")
    if chunk_size <= 0:
        raise ValueError("chunk_size müsbət olmalıdır")

    step = chunk_size - chunk_overlap
    spans = []
    i = 0
    text_len = len(text)

    while i < text_len:
        end = min(i + chunk_size, text_len)
        spans.append((i, end))
        if end == text_len:
            break
        i += step

    return spans


def chunk_documents(documents: list, chunk_size: int = DEFAULT_CHUNK_SIZE,
                     chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list:
    all_chunks = []
    chunk_counter = 0

    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        spans = sliding_window_chunk(text, chunk_size, chunk_overlap)

        for local_idx, (start, end) in enumerate(spans):
            all_chunks.append({
                "chunk_id": f"{source}::chunk_{local_idx}",
                "source": source,
                "chunk_index": local_idx,
                "start_char": start,
                "end_char": end,
                "text": text[start:end],
            })
            chunk_counter += 1

    return all_chunks
