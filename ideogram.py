"""Annotation-based Ideogram helper nodes."""

try:
    from .node_api import Int, String, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # allows quick local import tests outside package loading
    from node_api import Int, String, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


JSON_TEXT = String(multiline=True)
QUERY_TEXT = String(default="", multiline=True)
EMPTY_TEXT = String(default="", multiline=True)
AESTHETICS_TEXT = String(
    default="photorealistic editorial image, composition preserved from the source reference",
    multiline=True,
)
LIGHTING_TEXT = String(default="natural cinematic light matching the source composition", multiline=True)
PHOTO_TEXT = String(default="high quality realistic photograph", multiline=True)
MEDIUM_TEXT = String(default="photorealistic digital image", multiline=True)


def _json_load(text: str, fallback: object) -> object:
    import json

    try:
        return json.loads(text or "")
    except Exception:
        return fallback


@node("roka/ideogram/RK_Ideogram4JsonPromptComposer", returns=("json_prompt",))
def ideogram4_json_prompt_composer(
    elements_json: JSON_TEXT,
    high_level_description: EMPTY_TEXT = "",
    background: EMPTY_TEXT = "",
    aesthetics: AESTHETICS_TEXT = "photorealistic editorial image, composition preserved from the source reference",
    lighting: LIGHTING_TEXT = "natural cinematic light matching the source composition",
    photo: PHOTO_TEXT = "high quality realistic photograph",
    medium: MEDIUM_TEXT = "photorealistic digital image",
) -> str:
    """Compose a full Ideogram v4 JSON prompt from element JSON."""
    import json as jsonlib

    elements = _json_load(elements_json, [])
    source_prompt = elements if isinstance(elements, dict) else {}
    if isinstance(elements, dict):
        # Accept either a full prompt or a compositional_deconstruction object for convenience.
        if isinstance(elements.get("compositional_deconstruction"), dict):
            elements = elements["compositional_deconstruction"].get("elements", [])
        else:
            elements = elements.get("elements", [])
    if not isinstance(elements, list):
        elements = []

    if not high_level_description and isinstance(source_prompt.get("high_level_description"), str):
        high_level_description = source_prompt.get("high_level_description", "")

    prompt = {
        "high_level_description": str(high_level_description or ""),
        "style_description": {
            "aesthetics": str(aesthetics or ""),
            "lighting": str(lighting or ""),
            "photo": str(photo or ""),
            "medium": str(medium or ""),
        },
        "compositional_deconstruction": {
            "background": str(background or ""),
            "elements": elements,
        },
    }
    return jsonlib.dumps(prompt, indent=2, ensure_ascii=False)


@node("roka/ideogram/RK_IdeogramCrop", returns=("ideogram_json", "elements_json"))
def ideogram_crop(
    ideogram_json: JSON_TEXT,
    width: Int(min=1, default=1024),
    height: Int(min=1, default=1024),
    x1: Int(min=0, default=0),
    y1: Int(min=0, default=0),
    x2: Int(min=1, default=1024),
    y2: Int(min=1, default=1024),
) -> tuple[str, str]:
    """Crop Ideogram v4 element bboxes and return updated full JSON plus elements JSON."""
    import copy
    import json as jsonlib

    DISCARD_OUTSIDE_RATIO = 0.4

    prompt = _json_load(ideogram_json, {})
    if not isinstance(prompt, dict):
        prompt = {}

    out_prompt = copy.deepcopy(prompt)
    compositional = out_prompt.setdefault("compositional_deconstruction", {})
    if not isinstance(compositional, dict):
        compositional = {}
        out_prompt["compositional_deconstruction"] = compositional

    source_elements = compositional.get("elements", [])
    if not isinstance(source_elements, list):
        source_elements = []

    source_w = float(width)
    source_h = float(height)
    crop_x1 = float(x1)
    crop_y1 = float(y1)
    crop_x2 = float(x2)
    crop_y2 = float(y2)
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    def valid_bbox(box: object) -> bool:
        if not isinstance(box, list) or len(box) != 4:
            return False
        try:
            y1n, x1n, y2n, x2n = [float(v) for v in box]
        except Exception:
            return False
        return x2n > x1n and y2n > y1n

    def clamp_norm(value: float) -> int:
        return max(0, min(1000, round(value)))

    elements = []
    if source_w > 0 and source_h > 0 and crop_x1 >= 0 and crop_y1 >= 0 and crop_w > 0 and crop_h > 0:
        for element in source_elements:
            if not isinstance(element, dict) or not valid_bbox(element.get("bbox")):
                continue

            y1n, x1n, y2n, x2n = [float(v) for v in element["bbox"]]
            box_x1 = x1n / 1000.0 * source_w
            box_y1 = y1n / 1000.0 * source_h
            box_x2 = x2n / 1000.0 * source_w
            box_y2 = y2n / 1000.0 * source_h
            box_area = (box_x2 - box_x1) * (box_y2 - box_y1)
            if box_area <= 0:
                continue

            inside_x1 = max(box_x1, crop_x1)
            inside_y1 = max(box_y1, crop_y1)
            inside_x2 = min(box_x2, crop_x2)
            inside_y2 = min(box_y2, crop_y2)
            inside_w = max(0.0, inside_x2 - inside_x1)
            inside_h = max(0.0, inside_y2 - inside_y1)
            inside_area = inside_w * inside_h
            outside_ratio = 1.0 - (inside_area / box_area)
            if outside_ratio >= DISCARD_OUTSIDE_RATIO or inside_area <= 0:
                continue

            cropped = dict(element)
            cropped["bbox"] = [
                clamp_norm(((inside_y1 - crop_y1) / crop_h) * 1000.0),
                clamp_norm(((inside_x1 - crop_x1) / crop_w) * 1000.0),
                clamp_norm(((inside_y2 - crop_y1) / crop_h) * 1000.0),
                clamp_norm(((inside_x2 - crop_x1) / crop_w) * 1000.0),
            ]
            elements.append(cropped)

    compositional["elements"] = elements
    return (
        jsonlib.dumps(out_prompt, indent=2, ensure_ascii=False),
        jsonlib.dumps(elements, indent=2, ensure_ascii=False),
    )


@node("roka/sam3/RK_SceneGraphToIdeogram4Json", returns=("elements_json",))
def scene_graph_to_ideogram4_json(scenegraph: JSON_TEXT, query: QUERY_TEXT = "") -> str:
    """Convert selected scene graph subtrees into Ideogram v4 elements JSON."""
    import json as jsonlib
    import re

    nodes = _json_load(scenegraph, [])
    if not isinstance(nodes, list):
        nodes = []
    nodes = [node for node in nodes if isinstance(node, dict)]

    def valid_bbox(box: object) -> bool:
        return isinstance(box, list) and len(box) == 4

    def node_id(node: dict, fallback: int) -> object:
        return node.get("id", fallback)

    def node_label(node: dict) -> str:
        return str(node.get("label") or "").strip()

    def is_wrapper(node: dict) -> bool:
        label = node_label(node).lower()
        return label in {"foreground", "midground", "background"} and node.get("parent_id") is None

    node_by_id = {node_id(node, i): node for i, node in enumerate(nodes)}
    children: dict[object, list[object]] = {}
    roots: list[object] = []
    for i, node in enumerate(nodes):
        nid = node_id(node, i)
        parent_id = node.get("parent_id")
        if parent_id in node_by_id and parent_id != nid:
            children.setdefault(parent_id, []).append(nid)
        else:
            roots.append(nid)

    paths: dict[object, str] = {}
    order: list[object] = []

    def walk_paths(nid: object, prefix: list[str]) -> None:
        node = node_by_id.get(nid)
        if not isinstance(node, dict) or nid in paths:
            return
        path = prefix + [node_label(node)]
        paths[nid] = ".".join(part for part in path if part)
        order.append(nid)
        for child_id in children.get(nid, []):
            walk_paths(child_id, path)

    for root_id in roots:
        walk_paths(root_id, [])
    for nid in node_by_id:
        walk_paths(nid, [])

    patterns = [line.strip() for line in str(query or "").splitlines() if line.strip()]
    regexes = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    selected: set[object] = set()

    def select_subtree(nid: object) -> None:
        if nid in selected:
            return
        selected.add(nid)
        for child_id in children.get(nid, []):
            select_subtree(child_id)

    if regexes:
        for nid in order:
            path = paths.get(nid, "")
            if any(regex.search(path) for regex in regexes):
                select_subtree(nid)
    else:
        selected.update(order)

    all_boxes = [node.get("bbox") for node in nodes if valid_bbox(node.get("bbox"))]
    if not all_boxes:
        return jsonlib.dumps([], indent=2)

    scene_x1 = min(float(box[0]) for box in all_boxes)
    scene_y1 = min(float(box[1]) for box in all_boxes)
    scene_x2 = max(float(box[2]) for box in all_boxes)
    scene_y2 = max(float(box[3]) for box in all_boxes)
    scene_w = max(1.0, scene_x2 - scene_x1)
    scene_h = max(1.0, scene_y2 - scene_y1)

    def include_node(nid: object, node: dict) -> bool:
        return nid in selected and not is_wrapper(node) and valid_bbox(node.get("bbox"))

    def norm_x(value: object) -> int:
        scaled = (float(value) - scene_x1) / scene_w
        return max(0, min(1000, round(scaled * 1000)))

    def norm_y(value: object) -> int:
        scaled = (float(value) - scene_y1) / scene_h
        return max(0, min(1000, round(scaled * 1000)))

    elements = []
    for nid in order:
        item = node_by_id.get(nid, {})
        if not include_node(nid, item):
            continue
        x1, y1, x2, y2 = item.get("bbox")
        desc = str(item.get("caption") or item.get("desc") or item.get("label") or "item").strip() or "item"
        elements.append({
            "type": "obj",
            "bbox": [norm_y(y1), norm_x(x1), norm_y(y2), norm_x(x2)],
            "desc": desc,
        })

    return jsonlib.dumps(elements, indent=2, ensure_ascii=False)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
