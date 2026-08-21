from __future__ import annotations
import json, os, platform, random, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import scipy, sklearn, catboost, joblib, matplotlib


def resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "configs" / "project_config.json").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not resolve project root from current directory.")


def load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)


def json_default(value: Any):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def package_versions() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "catboost": catboost.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
    }


def ensure_output_directories(root: Path, project_config: dict) -> dict[str, Path]:
    paths = {}
    for key, rel in project_config["output_paths"].items():
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        paths[key] = path
    return paths


def relative_to_root(path: Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(root.resolve())).replace("\\", "/")
