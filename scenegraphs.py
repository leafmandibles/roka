"""Annotation-based scene graph helper nodes."""

try:
    from .node_api import InputSpec, String, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # allows quick local import tests outside package loading
    from node_api import InputSpec, String, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

MASK = InputSpec("MASK")
IMAGE = InputSpec("IMAGE")
JSON_TEXT = String(multiline=True)


def _json_load(text: str, fallback: object) -> object:
    import json

    try:
        return json.loads(text or "")
    except Exception:
        return fallback


def _mask_2d(mask: object):
    mask = mask > 0.5
    while mask.dim() > 2:
        mask = mask.any(dim=0)
    return mask


def _depth_2d(depth_map: object):
    depth = depth_map
    while depth.dim() > 2:
        if depth.shape[-1] in (1, 3, 4):
            depth = depth[..., 0]
        else:
            depth = depth[0]
    return depth.float()


def _bbox_from_mask(mask: object) -> list[int]:
    import torch

    ys, xs = torch.where(mask)
    if xs.numel() == 0 or ys.numel() == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _label_for(index: int, label_ranges: object) -> str:
    if isinstance(label_ranges, dict):
        for label, span in label_ranges.items():
            if isinstance(span, (list, tuple)) and len(span) >= 2 and int(span[0]) <= index < int(span[1]):
                return str(label)
    return "item"


def score_mask(mask: object, depth_map: object) -> dict[str, float | None]:
    import torch

    depth = _depth_2d(depth_map).to(mask.device)
    if depth.shape != mask.shape:
        depth = torch.nn.functional.interpolate(
            depth[None, None], size=tuple(mask.shape), mode="bilinear", align_corners=False
        )[0, 0]
    pixels = depth[mask]
    pixels = pixels[torch.isfinite(pixels)].float()
    if pixels.numel() == 0:
        return {"nearest_chunk": None}

    # Depth map is near-to-far: larger/brighter is nearer. Sort far -> near,
    # then read the actual pixel value at the closest ~12.5% boundary.
    sorted_pixels = torch.sort(pixels).values
    index = min(sorted_pixels.numel() - 1, int(sorted_pixels.numel() * 0.875))
    return {"nearest_chunk": float(sorted_pixels[index].item())}


@node("roka/sam3/RK_SAM3DepthSceneGraph", returns=("scenegraph",))
def sam3_depth_scene_graph(json: JSON_TEXT, masks: MASK, depth_map: IMAGE) -> str:
    """Build a SAM3 scene graph and attach nearest-chunk depth scores per mask."""
    import json as jsonlib
    import torch

    label_ranges = _json_load(json, {})
    count = int(masks.shape[0]) if masks is not None else 0
    mask_items = [_mask_2d(masks[i]) for i in range(count)]
    areas = [int(mask.sum().item()) for mask in mask_items]
    boxes = [_bbox_from_mask(mask) for mask in mask_items]

    nodes: list[dict[str, object]] = []
    for i, mask in enumerate(mask_items):
        parent_id = None
        best_area = 0
        for j, other in enumerate(mask_items):
            if i == j or areas[j] <= areas[i]:
                continue
            overlap = torch.logical_and(mask, other).sum().item()
            if overlap / max(1, areas[i]) > 0.6 and areas[j] > best_area:
                parent_id, best_area = j, areas[j]
        nodes.append({
            "id": i,
            "label": _label_for(i, label_ranges),
            "bbox": boxes[i],
            "depth_scores": score_mask(mask, depth_map),
            "parent_id": parent_id,
            "foreground": True,
        })
    return jsonlib.dumps(nodes, indent=2)


MODE = InputSpec(["relative", "absolute"], default="relative")


def _node_id(node: dict[str, object], fallback: int) -> int:
    try:
        return int(node.get("id", fallback))
    except Exception:
        return fallback


def _depth_score(node: dict[str, object]) -> float | None:
    scores = node.get("depth_scores")
    value = scores.get("nearest_chunk") if isinstance(scores, dict) else None
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _depth_bucket(score: float | None, mode: str, lo: float, hi: float) -> str:
    if score is None:
        return "background"
    if mode == "absolute":
        t = max(0.0, min(1.0, score))
    else:
        t = 1.0 if hi <= lo else (score - lo) / (hi - lo)
    if t >= 2.0 / 3.0:
        return "foreground"
    if t >= 1.0 / 3.0:
        return "midground"
    return "background"


def _clean_label(value: object) -> str:
    return str(value or "item").strip() or "item"


def _label_union(ids: list[int], node_by_id: dict[int, dict[str, object]]) -> str:
    labels: list[str] = []
    seen: set[str] = set()
    for nid in ids:
        label = _clean_label(node_by_id.get(nid, {}).get("label"))
        key = label.lower()
        if key not in seen:
            seen.add(key)
            labels.append(label)
    return ", ".join(labels) or "item"


def _bbox_union(ids: list[int], node_by_id: dict[int, dict[str, object]], masks: object) -> list[int] | None:
    boxes: list[list[int]] = []
    n_masks = int(masks.shape[0]) if masks is not None else 0
    for nid in ids:
        box = node_by_id.get(nid, {}).get("bbox")
        if isinstance(box, list) and len(box) == 4:
            boxes.append([int(v) for v in box])
        elif masks is not None and 0 <= nid < n_masks:
            boxes.append(_bbox_from_mask(_mask_2d(masks[nid])))
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


@node("roka/sam3/RK_SceneGraphDepthReducer", returns=("reduced_scenegraph",))
def scene_graph_depth_reducer(scenegraph: JSON_TEXT, masks: MASK, mode: MODE = "relative") -> str:
    """Bucket a depth-scored scene graph into foreground, midground, and background."""
    import json as jsonlib
    import torch

    nodes = _json_load(scenegraph, [])
    nodes = nodes if isinstance(nodes, list) else []
    node_by_id = {_node_id(node, i): node for i, node in enumerate(nodes) if isinstance(node, dict)}
    ordered_ids = list(node_by_id.keys())
    scores = [_depth_score(node_by_id[nid]) for nid in ordered_ids]
    valid_scores = [score for score in scores if score is not None]
    lo, hi = (min(valid_scores), max(valid_scores)) if valid_scores else (0.0, 1.0)

    children: dict[int, list[int]] = {}
    for nid in ordered_ids:
        try:
            parent = node_by_id[nid].get("parent_id")
            parent = int(parent) if parent is not None else None
        except Exception:
            parent = None
        if parent in node_by_id and parent != nid:
            children.setdefault(parent, []).append(nid)

    bucket_by_id = {nid: _depth_bucket(_depth_score(node_by_id[nid]), str(mode), lo, hi) for nid in ordered_ids}
    out: list[dict[str, object]] = []
    next_id = 0

    def add(label: str, parent_id: int | None, source_ids: list[int] | None = None) -> int:
        nonlocal next_id
        rid = next_id
        next_id += 1
        item: dict[str, object] = {"id": rid, "label": label, "parent_id": parent_id}
        if source_ids is not None:
            item.update({"bbox": _bbox_union(source_ids, node_by_id, masks), "source_ids": source_ids})
        out.append(item)
        return rid

    def descendants(nid: int, bucket: str) -> list[int]:
        found: list[int] = []
        def walk(current: int) -> None:
            if bucket_by_id.get(current) != bucket:
                return
            found.append(current)
            for child in children.get(current, []):
                walk(child)
        walk(nid)
        return found

    def grouped_overlaps(ids: list[int]) -> list[list[int]]:
        n_masks = int(masks.shape[0]) if masks is not None else 0
        valid = [nid for nid in ids if 0 <= nid < n_masks]
        if len(valid) < 2:
            return [[nid] for nid in ids]
        parent = {nid: nid for nid in ids}
        mask_by_id = {nid: _mask_2d(masks[nid]) for nid in valid}

        def find(nid: int) -> int:
            while parent[nid] != nid:
                parent[nid] = parent[parent[nid]]
                nid = parent[nid]
            return nid

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for offset, a in enumerate(valid):
            for b in valid[offset + 1:]:
                if torch.logical_and(mask_by_id[a], mask_by_id[b]).any().item():
                    union(a, b)
        groups: dict[int, list[int]] = {}
        for nid in ids:
            groups.setdefault(find(nid), []).append(nid)
        return list(groups.values())

    def emit_group(ids: list[int], parent_id: int, bucket: str) -> None:
        if len(ids) > 1:
            source_ids = sorted({sid for nid in ids for sid in descendants(nid, bucket)})
            add(_label_union(source_ids, node_by_id), parent_id, source_ids)
            return
        emit(ids[0], parent_id, bucket)

    def emit(nid: int, parent_id: int, bucket: str) -> None:
        new_id = add(_clean_label(node_by_id[nid].get("label")), parent_id, [nid])
        child_ids = [child for child in children.get(nid, []) if bucket_by_id.get(child) == bucket]
        for group in grouped_overlaps(child_ids):
            emit_group(group, new_id, bucket)

    for bucket in ("foreground", "midground", "background"):
        bucket_id = add(bucket, None)
        bucket_ids = [nid for nid in ordered_ids if bucket_by_id.get(nid) == bucket]
        roots = []
        for nid in bucket_ids:
            parent = node_by_id[nid].get("parent_id")
            try:
                parent = int(parent) if parent is not None else None
            except Exception:
                parent = None
            if parent not in bucket_ids:
                roots.append(nid)
        for group in grouped_overlaps(roots):
            emit_group(group, bucket_id, bucket)

    return jsonlib.dumps(out, indent=2)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
