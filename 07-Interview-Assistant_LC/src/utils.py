from pathlib import Path
from typing import List


def data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def load_texts_sample(limit: int = 3) -> List[str]:
    d = data_dir()
    if not d.exists():
        return []
    texts = []
    for p in d.rglob("*.txt"):
        try:
            texts.append(p.read_text(encoding="utf-8")[:1000])
        except Exception:
            continue
        if len(texts) >= limit:
            break
    return texts
