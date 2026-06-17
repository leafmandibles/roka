"""Private helper functions shared by Roka legacy nodes."""

def _rk_json_load(value, fallback=None):
    import json

    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return fallback
        return json.loads(text)
    return fallback


def _rk_json_dump(value):
    import json

    return json.dumps(value, indent=2, ensure_ascii=False)


def _rk_comfy_image_to_pil(image):
    import numpy as np
    from PIL import Image

    img = image[0].detach().cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def _rk_pil_to_comfy_image(pil_image):
    import numpy as np
    import torch

    arr = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _rk_sam3_load_module(rel_name, file_name):
    import sys
    import os
    import importlib.util
    import types

    sam3_root = os.path.dirname(os.path.abspath(__file__))
    package_name = "rk_sam3_external"
    nodes_name = package_name + ".nodes"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [sam3_root]
        sys.modules[package_name] = pkg
    if nodes_name not in sys.modules:
        nodes_pkg = types.ModuleType(nodes_name)
        nodes_pkg.__path__ = [sam3_root]
        sys.modules[nodes_name] = nodes_pkg

    full_name = nodes_name + "." + rel_name
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, os.path.join(sam3_root, file_name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def _rk_import_sam3_grounding():
    import sys

    for module in list(sys.modules.values()):
        mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
        if isinstance(mappings, dict) and "SAM3Grounding" in mappings and isinstance(mappings["SAM3Grounding"], type):
            return mappings["SAM3Grounding"], _rk_comfy_image_to_pil, _rk_pil_to_comfy_image
        cls = getattr(module, "SAM3Grounding", None)
        if isinstance(cls, type):
            return cls, _rk_comfy_image_to_pil, _rk_pil_to_comfy_image

    _rk_sam3_load_module("utils", "utils.py")
    _rk_sam3_load_module("sam3_model_patcher", "sam3_model_patcher.py")
    segmentation = _rk_sam3_load_module("segmentation", "segmentation.py")
    return segmentation.SAM3Grounding, _rk_comfy_image_to_pil, _rk_pil_to_comfy_image


def _rk_mask_bbox(mask):
    import torch

    ys, xs = torch.where(mask > 0.5)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _rk_import_sam3_loader():
    _rk_sam3_load_module("sam3_model_patcher", "sam3_model_patcher.py")
    load_model = _rk_sam3_load_module("load_model", "load_model.py")
    return load_model.LoadSAM3Model


def _rk_wordnet_parse_words(words):
    seen, parsed = set(), []
    for raw_word in str(words).split(","):
        word = _rk_wordnet_clean_label(raw_word)
        key = word.lower()
        if word and key not in seen:
            parsed.append(word)
            seen.add(key)
    return parsed


def _rk_wordnet_clean_label(value):
    import re
    return re.sub(r"\s+", " ", str(value).strip())


def _rk_wordnet_new_branch(label):
    return {"label": label, "children": {}}


def _rk_wordnet_insert_labels(root, labels):
    current = root
    for label in labels:
        label = _rk_wordnet_clean_label(label)
        if not label:
            continue
        key = label.lower()
        if key == current["label"].lower():
            continue
        if key not in current["children"]:
            current["children"][key] = _rk_wordnet_new_branch(label)
        current = current["children"][key]


def _rk_wordnet_synset_label(synset, show_synset_ids, show_definitions):
    name = synset.lemmas()[0].name().replace("_", " ")
    extras = []
    if show_synset_ids:
        extras.append(synset.name())
    if show_definitions:
        extras.append(synset.definition())
    return f"{name} ({'; '.join(extras)})" if extras else name


def _rk_wordnet_render_branch(root):
    lines = [root["label"]]
    children = _rk_wordnet_sorted_children(root)
    for index, child in enumerate(children):
        _rk_wordnet_append_rendered_branch(lines, child, "", index == len(children) - 1)
    return "\n".join(lines)


def _rk_wordnet_append_rendered_branch(lines, branch, prefix, is_last):
    connector = "└── " if is_last else "├── "
    lines.append(f"{prefix}{connector}{branch['label']}")
    child_prefix = prefix + ("    " if is_last else "│   ")
    children = _rk_wordnet_sorted_children(branch)
    for index, child in enumerate(children):
        _rk_wordnet_append_rendered_branch(lines, child, child_prefix, index == len(children) - 1)


def _rk_wordnet_sorted_children(branch):
    return sorted(branch["children"].values(), key=lambda child: child["label"].lower())


def _rk_scene_caption(label, enricher=None):
    label = str(label or "item").strip() or "item"
    if isinstance(enricher, dict):
        enriched = enricher.get(label) or enricher.get(label.lower())
        if isinstance(enriched, list) and enriched:
            value = str(enriched[0]).strip()
            if value:
                return value
        if isinstance(enriched, str) and enriched.strip():
            return enriched.strip()
    return label


def _rk_parse_aspect_ratio(value):
    import re

    text = str(value or "1:1").strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)", text)
    if not match:
        raise ValueError(f"Invalid aspect ratio '{value}'. Use W:H, e.g. 4:3.")
    w = float(match.group(1))
    h = float(match.group(2))
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid aspect ratio '{value}'. Values must be positive.")
    return w, h


def _rk_snap_int(value, snap):
    snap = max(1, int(snap or 1))
    return max(snap, int(round(float(value) / snap) * snap))


def _rk_aspect_label(width, height):
    import math
    from fractions import Fraction

    width = max(1, int(width or 1))
    height = max(1, int(height or 1))
    gcd = math.gcd(width, height)
    simple_w, simple_h = width // gcd, height // gcd
    if simple_w <= 100 and simple_h <= 100:
        return f"{simple_w}:{simple_h}"
    frac = Fraction(width, height).limit_denominator(100)
    return f"{frac.numerator}:{frac.denominator}"


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

