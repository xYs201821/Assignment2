from __future__ import annotations

import pickle
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


_DATASET_ALIASES = {
    "jsb": "jsb_chorales",
    "jsb_chorales": "jsb_chorales",
    "jsb-chorales": "jsb_chorales",
    "musedata": "musedata",
    "muse_data": "musedata",
}

_DATASET_FILENAMES = {
    "jsb_chorales": [
        "JSB Chorales.pickle",
        "JSB Chorales.pkl",
        "JSB_Chorales.pickle",
        "jsb_chorales.pickle",
        "jsb_chorales.pkl",
        "jsb-chorales-16th.pkl",
    ],
    "musedata": [
        "MuseData.pickle",
        "MuseData.pkl",
        "musedata.pickle",
        "musedata.pkl",
    ],
}

_DATASET_URLS = {
    "jsb_chorales": [
        "https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/JSB%20Chorales.pickle",
        "https://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.pickle",
        "http://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.pickle",
        "https://raw.githubusercontent.com/pyro-ppl/pyro/dev/examples/polyphonic_data/JSB%20Chorales.pickle",
    ],
    "musedata": [
        "https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/MuseData.pickle",
        "https://www-etud.iro.umontreal.ca/~boulanni/MuseData.pickle",
        "http://www-etud.iro.umontreal.ca/~boulanni/MuseData.pickle",
        "https://raw.githubusercontent.com/pyro-ppl/pyro/dev/examples/polyphonic_data/MuseData.pickle",
    ],
}

_SPLIT_ALIASES = {
    "train": ("train", "training"),
    "valid": ("valid", "validation", "val", "dev"),
    "test": ("test", "testing"),
}


def _canonical_dataset_name(dataset: str) -> str:
    key = str(dataset).strip().lower()
    if key not in _DATASET_ALIASES:
        supported = ", ".join(sorted(_DATASET_ALIASES))
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported: {supported}")
    return _DATASET_ALIASES[key]


def _find_existing_file(root: Path, candidates: Iterable[str]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.exists() and path.is_file():
            return path
    return None


def _try_download(urls: List[str], target: Path) -> Path:
    last_error: Exception | None = None
    target.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            urllib.request.urlretrieve(url, target)  # noqa: S310
            return target
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
    attempted = "\n  - ".join(urls)
    raise RuntimeError(
        "Failed to download dataset.\n"
        f"Target path: {target}\n"
        f"Attempted URLs:\n  - {attempted}\n"
        "You can manually place the dataset file at the target path."
    ) from last_error


def _locate_dataset_file(dataset_name: str, root: Path, download: bool) -> Path:
    candidates = _DATASET_FILENAMES[dataset_name]
    existing = _find_existing_file(root, candidates)
    if existing is not None:
        return existing
    if not download:
        names = ", ".join(candidates)
        raise FileNotFoundError(
            f"Dataset file for '{dataset_name}' not found in {root}. "
            f"Expected one of: {names}"
        )
    target = root / candidates[0]
    return _try_download(_DATASET_URLS[dataset_name], target)


def _load_raw_payload(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k].tolist() for k in data.files}
    with path.open("rb") as f:
        payload = pickle.load(f, encoding="latin1")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload).__name__}")
    return payload


def _resolve_split(raw: Dict[str, Any], split: str) -> List[Any]:
    split_key = str(split).strip().lower()
    matched = _SPLIT_ALIASES.get(split_key, (split_key,))
    for key in matched:
        if key in raw:
            value = raw[key]
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, list):
                return value
            return list(value)
    available = ", ".join(sorted(raw.keys()))
    raise KeyError(f"Split '{split}' not found. Available keys: {available}")


def _notes_to_roll(seq: Any, obs_dim: int) -> np.ndarray:
    try:
        arr = np.asarray(seq)
    except ValueError:
        arr = np.asarray(seq, dtype=object)
    if arr.ndim == 2 and arr.shape[-1] == obs_dim:
        return (arr > 0).astype(np.float32)

    steps: List[np.ndarray] = []
    for notes in seq:
        step = np.zeros((obs_dim,), dtype=np.float32)
        for note in notes:
            note_int = int(note)
            idx = note_int - 21 if 21 <= note_int <= 108 else note_int
            if 0 <= idx < obs_dim:
                step[idx] = 1.0
        steps.append(step)
    if not steps:
        return np.zeros((0, obs_dim), dtype=np.float32)
    return np.stack(steps, axis=0)


def _pad_or_truncate(roll: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    length = int(roll.shape[0])
    out = np.zeros((seq_len, roll.shape[-1]), dtype=np.float32)
    mask = np.zeros((seq_len,), dtype=np.float32)
    use = min(length, seq_len)
    if use > 0:
        out[:use] = roll[:use]
        mask[:use] = 1.0
    return out, mask


def load_polyphonic_split(
    dataset: str,
    root: str | Path,
    *,
    split: str = "train",
    seq_len: int = 150,
    obs_dim: int = 88,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    dataset_name = _canonical_dataset_name(dataset)
    root_path = Path(root)
    file_path = _locate_dataset_file(dataset_name, root_path, download=bool(download))
    payload = _load_raw_payload(file_path)
    raw_sequences = _resolve_split(payload, split=split)

    y_list: List[np.ndarray] = []
    m_list: List[np.ndarray] = []
    for seq in raw_sequences:
        roll = _notes_to_roll(seq, obs_dim=obs_dim)
        y_pad, m_pad = _pad_or_truncate(roll, seq_len=seq_len)
        y_list.append(y_pad)
        m_list.append(m_pad)

    if not y_list:
        raise ValueError(f"No sequences found for split '{split}' in {file_path}")

    y = np.stack(y_list, axis=0).astype(np.float32)
    mask = np.stack(m_list, axis=0).astype(np.float32)
    meta = {
        "dataset": dataset_name,
        "file": str(file_path),
        "split": split,
        "num_sequences": int(y.shape[0]),
        "seq_len": int(seq_len),
        "obs_dim": int(obs_dim),
    }
    return y, mask, meta


__all__ = [
    "load_polyphonic_split",
]
