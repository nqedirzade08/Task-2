import os
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = (".txt", ".pdf")


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(folder_path: str) -> list:
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Qovluq tapılmadı: {folder_path}")

    documents = []
    for filename in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        full_path = os.path.join(folder_path, filename)

        try:
            if ext == ".txt":
                text = load_txt(full_path)
            elif ext == ".pdf":
                text = load_pdf(full_path)
            else:
                continue
        except Exception as e:
            print(f"[Xəbərdarlıq] '{filename}' oxuna bilmədi: {type(e).__name__}: {e}")
            continue

        text = text.strip()
        if not text:
            print(f"[Xəbərdarlıq] '{filename}' boşdur, keçilir.")
            continue

        documents.append({"source": filename, "text": text})

    return documents
