"""Annotation-based cache helper nodes."""

try:
    from .node_api import Any, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # allows quick local import tests outside package loading
    from node_api import Any, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def _rk_roka_cache_root():
    import os

    try:
        import folder_paths
        output_dir = folder_paths.get_output_directory()
    except Exception:
        output_dir = os.path.join(os.getcwd(), "output")
    return os.path.join(output_dir, "roka_cache")


def _rk_cache_safe_part(value, name):
    import os

    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    if text in {".", ".."}:
        raise ValueError(f"{name} must not be a relative path segment")
    if os.path.basename(text) != text or "/" in text or "\\" in text:
        raise ValueError(f"{name} must be a plain filename, not a path: {text!r}")
    return text


def _rk_cache_path(hashid, label=None):
    import os

    safe_hashid = _rk_cache_safe_part(hashid, "hashid")
    root = _rk_roka_cache_root()
    if label is None:
        return os.path.join(root, safe_hashid)
    safe_label = _rk_cache_safe_part(label, "label")
    return os.path.join(root, safe_hashid, safe_label)


@node("roka/cache/RK_HashFile", returns=("hashid",))
def hash_file(file_path: str = "") -> str:
    import hashlib
    import os

    path = str(file_path or "").strip()
    if not path:
        raise ValueError("file_path must not be empty")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@node("roka/cache/RK_HashCache", returns=("data",))
def hash_cache(hashid: str = "", *, data: Any, label: str = "data") -> Any:
    import os
    import pickle

    path = _rk_cache_path(hashid, label)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


@node("roka/cache/RK_CacheGet", returns=("data",))
def cache_get(hashid: str = "", label: str = "data") -> Any:
    import pickle

    path = _rk_cache_path(hashid, label)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


@node("roka/cache/RK_CacheInfo", returns=("labels_json",))
def cache_info(hashid: str = "") -> str:
    import json
    import os

    cache_dir = _rk_cache_path(hashid)
    if not os.path.isdir(cache_dir):
        return "[]"
    labels = sorted(name for name in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, name)))
    return json.dumps(labels, ensure_ascii=False)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
