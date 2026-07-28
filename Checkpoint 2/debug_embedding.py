import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key=os.getenv("NVIDIA_API_KEY"),
)

CANDIDATES = [
    ("baai/bge-m3", {}),
    ("nvidia/nv-embedqa-e5-v5", {"extra_body": {"input_type": "passage", "truncate": "END"}}),
    ("nvidia/llama-3.2-nv-embedqa-1b-v1", {"extra_body": {"input_type": "passage", "truncate": "END"}}),
    ("snowflake/arctic-embed-l", {}),
    ("nvidia/nv-embed-v1", {"extra_body": {"input_type": "passage", "truncate": "END"}}),
]

for model_id, kwargs in CANDIDATES:
    print(f"--- Test: {model_id} ---")
    try:
        response = client.embeddings.create(
            model=model_id,
            input="Bu bir test cümləsidir.",
            **kwargs,
        )
        print(f"  UĞURLU! Vektor ölçüsü: {len(response.data[0].embedding)}")
    except Exception as e:
        print(f"  XƏTA: {type(e).__name__}: {e}")
    print()
