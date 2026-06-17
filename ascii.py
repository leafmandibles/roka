"""Annotation-based ASCII rendering nodes."""

try:
    from .node_api import String, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # allows quick local import tests outside package loading
    from node_api import String, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


JSON_TEXT = String(multiline=True)


def _json_load(text: str, fallback: object) -> object:
    import json

    try:
        return json.loads(text or "")
    except Exception:
        return fallback


@node("roka/sam3/RK_SceneGraphAsciiRenderer", returns=("ascii",))
def scene_graph_ascii_renderer(scenegraph: JSON_TEXT) -> str:
    nodes = _json_load(scenegraph, [])
    if not isinstance(nodes, list):
        nodes = []

    node_by_id = {node.get("id"): node for node in nodes if isinstance(node, dict)}
    children = {}
    roots = []

    for node_id, node in node_by_id.items():
        parent = node.get("parent_id")
        if parent is None or parent not in node_by_id or parent == node_id:
            roots.append(node_id)
        else:
            children.setdefault(parent, []).append(node_id)

    for child_list in children.values():
        child_list.sort()
    roots.sort()

    def node_label(node_id):
        node = node_by_id.get(node_id, {"id": node_id, "label": "item"})
        label = node.get("caption") or node.get("desc") or node.get("label") or "item"
        return f"{node_id}: {label}"

    lines = []
    visited = set()

    def walk(node_id, prefix="", is_last=True):
        connector = "└── " if is_last else "├── "
        if node_id in visited:
            lines.append(f"{prefix}{connector}{node_label(node_id)} ↩")
            return
        visited.add(node_id)
        lines.append(f"{prefix}{connector}{node_label(node_id)}")
        next_prefix = prefix + ("    " if is_last else "│   ")
        child_list = children.get(node_id, [])
        for child_pos, child_id in enumerate(child_list):
            walk(child_id, next_prefix, child_pos == len(child_list) - 1)

    for root_pos, root_id in enumerate(roots):
        walk(root_id, "", root_pos == len(roots) - 1)
    for node_id in sorted(node_by_id):
        if node_id not in visited:
            walk(node_id, "", True)

    return "SceneGraph\n" + "\n".join(lines) if lines else "SceneGraph\n(empty)"


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
