"""Annotation-based cache helper nodes."""

try:
    from .node_api import Any, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # allows quick local import tests outside package loading
    from node_api import Any, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


_CACHE_MAGIC = b"RK_CACHE_V1\n"
_ALLOWED_DATA_TYPES = {"any", "IMAGE", "MASK", "LATENT"}


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


def _canonical_json_bytes(value):
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _pack_cache_blob(descriptor, payload):
    meta = _canonical_json_bytes(descriptor)
    return _CACHE_MAGIC + len(meta).to_bytes(8, "big") + meta + payload


def _unpack_cache_blob(blob):
    import json

    if not blob.startswith(_CACHE_MAGIC):
        raise ValueError("Unsupported cache blob format")
    offset = len(_CACHE_MAGIC)
    meta_len = int.from_bytes(blob[offset:offset + 8], "big")
    offset += 8
    meta = json.loads(blob[offset:offset + meta_len].decode("utf-8"))
    payload = blob[offset + meta_len:]
    return meta, payload


def _is_torch_tensor(value):
    try:
        import torch
    except Exception:
        return False
    return isinstance(value, torch.Tensor)


def _is_safetensors_saveable(value):
    return isinstance(value, dict) and all(isinstance(k, str) and _is_torch_tensor(v) for k, v in value.items())


def _serialize_cache_value(data, data_type="any"):
    dtype = str(data_type or "any").strip() or "any"
    if dtype not in _ALLOWED_DATA_TYPES:
        raise ValueError(f"data_type must be one of {sorted(_ALLOWED_DATA_TYPES)}")

    if isinstance(data, str):
        payload = data.encode("utf-8")
        return _pack_cache_blob({"version": 1, "kind": "STRING", "encoding": "utf-8"}, payload)

    if isinstance(data, bytes):
        return _pack_cache_blob({"version": 1, "kind": "BYTES"}, data)

    tensor_map = None
    unwrap_key = None
    if _is_torch_tensor(data):
        unwrap_key = "data"
        tensor_map = {unwrap_key: data}
    elif _is_safetensors_saveable(data):
        tensor_map = data

    if tensor_map is not None:
        from safetensors.torch import save

        payload = save({key: tensor.detach().cpu().contiguous() for key, tensor in sorted(tensor_map.items())})
        return _pack_cache_blob(
            {
                "version": 1,
                "kind": "TENSORS",
                "format": "safetensors",
                "data_type": dtype,
                "unwrap_key": unwrap_key,
            },
            payload,
        )

    raise TypeError("RK cache only supports str, bytes, torch.Tensor, or dict[str, torch.Tensor]")


def _deserialize_cache_value(blob):
    descriptor, payload = _unpack_cache_blob(blob)
    kind = descriptor.get("kind")

    if kind == "STRING":
        return payload.decode(descriptor.get("encoding", "utf-8"))
    if kind == "BYTES":
        return payload
    if kind == "TENSORS" and descriptor.get("format") == "safetensors":
        from safetensors.torch import load

        tensors = load(payload)
        unwrap_key = descriptor.get("unwrap_key")
        data_type = descriptor.get("data_type", "any")
        if data_type in {"IMAGE", "MASK"} and unwrap_key:
            return tensors[unwrap_key]
        if unwrap_key and data_type == "any":
            return tensors[unwrap_key]
        return tensors

    raise ValueError(f"Unsupported cache blob kind: {kind!r}")


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
def hash_cache(hashid: str = "", *, data: Any, label: str = "data", data_type: str = "any") -> Any:
    import os

    path = _rk_cache_path(hashid, label)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(_serialize_cache_value(data, data_type=data_type))
    return data


@node("roka/cache/RK_HashCacheEmitHashId", returns=("data", "hashid"), display_name="RK HashCache Emit Hash Id")
def hash_cache_emit_hash_id(*, data: Any, label: str = "data", data_type: str = "any") -> tuple[Any, str]:
    import hashlib
    import os

    blob = _serialize_cache_value(data, data_type=data_type)
    hashid = hashlib.sha256(blob).hexdigest()
    path = _rk_cache_path(hashid, label)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(blob)
    return data, hashid


@node("roka/cache/RK_CacheGet", returns=("data",))
def cache_get(hashid: str = "", label: str = "data") -> Any:
    path = _rk_cache_path(hashid, label)
    with open(path, "rb") as f:
        return _deserialize_cache_value(f.read())


@node("roka/cache/RK_CacheExists", returns=("exists",))
def cache_exists(hashid: str = "", label: str = "data") -> bool:
    import os

    path = _rk_cache_path(hashid, label)
    return os.path.isfile(path)


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
