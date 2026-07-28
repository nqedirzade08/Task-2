import os
import time
from dotenv import load_dotenv
from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    AuthenticationError,
    BadRequestError,
)

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "baai/bge-m3")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

if not API_KEY:
    raise ValueError(
        "NVIDIA_API_KEY tapılmadı. Zəhmət olmasa .env faylını yaradın "
        "və içinə NVIDIA_API_KEY=sizin_key_iniz yazın."
    )

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60, max_retries=0)

RETRYABLE_EXCEPTIONS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def call_with_retry(request_fn, max_attempts: int = 4, base_delay: float = 1.5):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return request_fn(), None
        except (AuthenticationError, BadRequestError) as e:
            return None, f"Xəta (təkrar cəhd edilmir): {type(e).__name__}: {e}"
        except RETRYABLE_EXCEPTIONS as e:
            last_error = e
            if attempt == max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            print(f"[Xəbərdarlıq] {type(e).__name__} baş verdi, {delay:.1f} saniyədən sonra "
                  f"{attempt}/{max_attempts - 1} təkrar cəhd edilir...")
            time.sleep(delay)
        except Exception as e:
            return None, f"Gözlənilməz xəta: {type(e).__name__}: {e}"

    return None, f"Bütün cəhdlər uğursuz oldu: {type(last_error).__name__}: {last_error}"


ASYMMETRIC_MODEL_HINTS = ("embedqa", "nv-embed", "e5")


def model_needs_input_type() -> bool:
    return any(hint in EMBEDDING_MODEL.lower() for hint in ASYMMETRIC_MODEL_HINTS)


def embed_batch(texts: list, input_type: str = "passage") -> list:
    def request_fn():
        kwargs = {"model": EMBEDDING_MODEL, "input": texts}
        if model_needs_input_type():
            kwargs["extra_body"] = {"input_type": input_type, "truncate": "END"}
        response = client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]

    result, error = call_with_retry(request_fn)
    if error:
        raise RuntimeError(error)
    return result


def embed_chunks(chunks: list, input_type: str = "passage") -> list:
    embedded = []
    total = len(chunks)

    for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = chunks[batch_start:batch_start + EMBEDDING_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        vectors = embed_batch(texts, input_type=input_type)

        for chunk, vector in zip(batch, vectors):
            embedded.append({**chunk, "embedding": vector, "embedding_dim": len(vector)})

        print(f"[Embedding] {min(batch_start + EMBEDDING_BATCH_SIZE, total)}/{total} chunk emb edildi")

    return embedded


def embed_query(text: str) -> list:
    vectors = embed_batch([text], input_type="query")
    return vectors[0]
