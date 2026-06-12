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


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
