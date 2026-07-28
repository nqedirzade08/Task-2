import json
import re


def extract_json_candidate(raw_text: str):
    if raw_text is None:
        return None, "Cavab boşdur (None)."

    candidate = raw_text.strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
    candidate = re.sub(r"\s*```$", "", candidate)

    try:
        return json.loads(candidate), None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), None
        except json.JSONDecodeError as e:
            return None, f"JSON tapıldı amma parse xətası: {e}"

    return None, "Mətndə JSON obyekti tapılmadı."


def validate_citation_schema(data) -> tuple:
    if not isinstance(data, dict):
        return False, f"Nəticə dict deyil, tipi: {type(data).__name__}"

    if "answer" not in data:
        return False, "Tələb olunan sahə çatışmır: 'answer'"
    if not isinstance(data["answer"], str) or not data["answer"].strip():
        return False, "'answer' sahəsi boş və ya yanlış tipdədir"

    if "sources" not in data:
        return False, "Tələb olunan sahə çatışmır: 'sources'"
    if not isinstance(data["sources"], list):
        return False, "'sources' sahəsi siyahı olmalıdır"
    if not all(isinstance(s, str) for s in data["sources"]):
        return False, "'sources' siyahısındakı bütün elementlər mətn (string) olmalıdır"

    return True, None


def resolve_citations(data: dict, label_map: dict) -> dict:
    resolved = []
    invalid_labels = []

    for label in data.get("sources", []):
        if label in label_map:
            hit = label_map[label]
            meta = hit["metadata"]
            resolved.append({
                "label": label,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "distance": hit["distance"],
            })
        else:
            invalid_labels.append(label)

    return {
        "answer": data["answer"],
        "citations": resolved,
        "invalid_labels": invalid_labels,
    }


def format_answer_with_citations(resolved: dict) -> str:
    lines = [resolved["answer"]]

    if resolved["citations"]:
        lines.append("")
        lines.append("Mənbələr:")
        for c in resolved["citations"]:
            lines.append(f"  - {c['source']} (chunk {c['chunk_index']}, məsafə={c['distance']:.4f})")

    if resolved["invalid_labels"]:
        lines.append("")
        lines.append(f"[Xəbərdarlıq] Model mövcud olmayan etiketlərə istinad etdi: {resolved['invalid_labels']}")

    return "\n".join(lines)
