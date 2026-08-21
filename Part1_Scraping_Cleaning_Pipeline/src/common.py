from __future__ import annotations
from pathlib import Path
import json, random, re
import numpy as np


def project_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    if p.name == "notebooks":
        return p.parent
    if (p / "configs" / "project_config.json").exists():
        return p
    for parent in p.parents:
        if (parent / "configs" / "project_config.json").exists():
            return parent
    return p


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class PersianTools:
    """Uses Hazm when installed; otherwise a minimal smoke-test fallback."""
    def __init__(self):
        self.backend = "fallback"
        try:
            import hazm
            self.normalizer = hazm.Normalizer()
            self.tokenizer = hazm.WordTokenizer()
            self.stemmer = hazm.Stemmer()
            self.backend = "hazm"
        except Exception:
            self.normalizer = None
            self.tokenizer = None
            self.stemmer = None

    def normalize(self, text) -> str:
        if text is None:
            return ""
        s = str(text)
        if self.backend == "hazm":
            return self.normalizer.normalize(s)
        table = str.maketrans({"ي":"ی", "ك":"ک", "ة":"ه", "ۀ":"ه"})
        s = s.translate(table).replace("\u200f", "").replace("\u200e", "")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def tokenize(self, text) -> list[str]:
        s = self.normalize(text)
        if self.backend == "hazm":
            return self.tokenizer.tokenize(s)
        return re.findall(r"[\u0600-\u06FF\u200c]+", s)

    def stem(self, token: str) -> str:
        if self.backend == "hazm":
            return self.stemmer.stem(token)
        return token
