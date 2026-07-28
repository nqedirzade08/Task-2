import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key=os.getenv("NVIDIA_API_KEY"),
)

try:
    models = client.models.list()
    print("Bütün mövcud modellər (embed/bge açar sözü ilə süzülmüş):\n")
    found = False
    for m in models.data:
        if "embed" in m.id.lower() or "bge" in m.id.lower():
            print(" -", m.id)
            found = True
    if not found:
        print("Heç bir embedding modeli tapılmadı. Bütün model siyahısı:\n")
        for m in models.data:
            print(" -", m.id)
except Exception as e:
    print("XƏTA TİPİ:", type(e).__name__)
    print("XƏTA MƏTNİ:", e)
