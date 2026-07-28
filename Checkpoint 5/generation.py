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
CHAT_MODEL = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-pro")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120, max_retries=0)

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


def generate_answer(messages: list) -> str:
    def request_fn():
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            top_p=0.9,
            max_tokens=500,
            extra_body={"chat_template_kwargs": {"thinking": False}},
            stream=False,
        )
        return completion.choices[0].message.content

    result, error = call_with_retry(request_fn)
    if error:
        return f"[Xəta] {error}"
    return result


def generate_citation_answer(messages: list, label_map: dict, max_json_attempts: int = 3) -> dict:
    from citation import extract_json_candidate, validate_citation_schema, resolve_citations

    current_messages = messages
    raw_result = ""
    last_reason = ""

    for attempt in range(1, max_json_attempts + 1):

        def request_fn():
            completion = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=current_messages,
                temperature=0.2,
                top_p=0.9,
                max_tokens=500,
                extra_body={"chat_template_kwargs": {"thinking": False}},
                stream=False,
            )
            return completion.choices[0].message.content

        raw_result, error = call_with_retry(request_fn)
        if error:
            return {"error": error}

        data, parse_error = extract_json_candidate(raw_result)
        if parse_error:
            last_reason = f"Parsing xətası: {parse_error}"
        else:
            is_valid, validation_error = validate_citation_schema(data)
            if is_valid:
                return resolve_citations(data, label_map)
            last_reason = f"Validasiya xətası: {validation_error}"

        print(f"[Xəbərdarlıq] Çıxış korrupt/etibarsızdır (cəhd {attempt}/{max_json_attempts}): {last_reason}")

        current_messages = messages + [
            {"role": "assistant", "content": raw_result},
            {"role": "user", "content": (
                f"Bu format YANLIŞ idi ({last_reason}). Cavabın YALNIZ "
                '{"answer": "...", "sources": [...]} formatında olmalıdır, başqa heç nə.'
            )},
        ]

    return {"error": f"{max_json_attempts} cəhddən sonra da düzgün/etibarlı JSON alına bilmədi.",
            "son_sebeb": last_reason, "son_xam_cavab": raw_result}
