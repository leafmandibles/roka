"""Annotation-based Ideogram helper nodes."""

try:
    from .node_api import String, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # allows quick local import tests outside package loading
    from node_api import String, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


JSON_TEXT = String(multiline=True)
QUERY_TEXT = String(default="", multiline=True)


def _json_load(text: str, fallback: object) -> object:
    import json

    try:
        return json.loads(text or "")
    except Exception:
        return fallback


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
