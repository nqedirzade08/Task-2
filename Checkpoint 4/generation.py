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
