"""Annotation-based structured output helper nodes."""

try:
    from .node_api import String, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # allows quick local import tests outside package loading
    from node_api import String, node, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

JSON_TEXT = String(multiline=True)
RULES_TEXT = String(
    multiline=True,
    default=".*foreground.*.face: a smiling beautiful face\n.*.wall: a fuchsia blue wall\nbackground.*.lamp: an {orange|blue} lamp",
)


def _json_load(text: str, fallback: object) -> object:
    import json

    try:
        return json.loads(text or "")
    except Exception:
        return fallback


def _json_dump(value: object) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False)


def _node_id(node: dict[str, object], fallback: int) -> object:
    return node.get("id", fallback)


def _node_label(node: dict[str, object]) -> str:
    return str(node.get("label") or node.get("caption") or node.get("desc") or "").strip()


def _parse_rules(query: str) -> list[tuple[int, str, str]]:
    rules: list[tuple[int, str, str]] = []
    for line_no, raw_line in enumerate((query or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        pattern, replacement = line.split(":", 1)
        pattern, replacement = pattern.strip(), replacement.strip()
        if pattern and replacement:
            rules.append((line_no, pattern, replacement))
    return rules


def _expand_choices(text: str, rng: object) -> str:
    import re

    def choose(match: re.Match[str]) -> str:
        options = [part.strip() for part in match.group(1).split("|") if part.strip()]
        return rng.choice(options) if options else match.group(0)

    return re.sub(r"\{([^{}|]+(?:\|[^{}|]+)+)\}", choose, text)


@node("roka/structured/RK_SceneGraphRegex", returns=("scenegraph", "summary"))
def scene_graph_regex(scenegraph: JSON_TEXT, query: RULES_TEXT) -> tuple[str, str]:
    """Apply path-aware regex replacements to leaf nodes in a scene graph."""
    import copy
    import random
    import re

    rng = random.Random(0)
    nodes = _json_load(scenegraph, [])
    if not isinstance(nodes, list):
        return (_json_dump(nodes), "Input is not a scenegraph list")

    out = [copy.deepcopy(node) for node in nodes if isinstance(node, dict)]
    node_by_id = {_node_id(node, i): node for i, node in enumerate(out)}
    children: dict[object, list[object]] = {}
    roots: list[object] = []
    for i, node in enumerate(out):
        node_id = _node_id(node, i)
        parent_id = node.get("parent_id")
        if parent_id in node_by_id and parent_id != node_id:
            children.setdefault(parent_id, []).append(node_id)
        else:
            roots.append(node_id)
    for child_ids in children.values():
        child_ids.sort(key=str)
    roots.sort(key=str)

    paths: dict[object, str] = {}

    def walk(node_id: object, prefix: list[str]) -> None:
        node = node_by_id.get(node_id)
        if not isinstance(node, dict) or node_id in paths:
            return
        path = prefix + [_node_label(node)]
        paths[node_id] = ".".join(part for part in path if part)
        for child_id in children.get(node_id, []):
            walk(child_id, path)

    for root_id in roots:
        walk(root_id, [])
    for node_id in node_by_id:
        walk(node_id, [])

    rules = [(line_no, re.compile(pattern, re.IGNORECASE), pattern, replacement) for line_no, pattern, replacement in _parse_rules(query)]
    applied: list[str] = []
    for i, node in enumerate(out):
        node_id = _node_id(node, i)
        if children.get(node_id):
            continue
        path = paths.get(node_id, "")
        for line_no, regex, pattern, replacement in rules:
            if not regex.search(path):
                continue
            text = _expand_choices(replacement, rng)
            node.setdefault("original_label", node.get("label"))
            node["label"] = text
            node["caption"] = text
            node["desc"] = text
            applied.append(f"line {line_no} {pattern!r} -> id {node_id} ({path})")
            break

    summary = f"Applied {len(applied)} replacement(s) from {len(rules)} rule(s)"
    if applied:
        summary += "\n" + "\n".join(applied)
    return (_json_dump(out), summary)


@node("roka/structured/RK_SceneGraphFlatten", returns=("text",))
def scene_graph_flatten(scenegraph: JSON_TEXT, query: str = ".*") -> str:
    """Return comma-separated labels from subtrees whose path matches a regex."""
    import re

    nodes = _json_load(scenegraph, [])
    if not isinstance(nodes, list):
        return ""

    node_by_id = {_node_id(node, i): node for i, node in enumerate(nodes) if isinstance(node, dict)}
    children: dict[object, list[object]] = {}
    roots: list[object] = []
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_id = _node_id(node, i)
        parent_id = node.get("parent_id")
        if parent_id in node_by_id and parent_id != node_id:
            children.setdefault(parent_id, []).append(node_id)
        else:
            roots.append(node_id)

    paths: dict[object, str] = {}
    order: list[object] = []

    def walk_paths(node_id: object, prefix: list[str]) -> None:
        node = node_by_id.get(node_id)
        if not isinstance(node, dict) or node_id in paths:
            return
        path = prefix + [_node_label(node)]
        paths[node_id] = ".".join(part for part in path if part)
        order.append(node_id)
        for child_id in children.get(node_id, []):
            walk_paths(child_id, path)

    for root_id in roots:
        walk_paths(root_id, [])
    for node_id in node_by_id:
        walk_paths(node_id, [])

    regex = re.compile(query or ".*", re.IGNORECASE)
    labels: list[str] = []
    emitted: set[object] = set()

    def emit_subtree(node_id: object) -> None:
        if node_id in emitted:
            return
        emitted.add(node_id)
        label = _node_label(node_by_id.get(node_id, {}))
        if label:
            labels.append(label)
        for child_id in children.get(node_id, []):
            emit_subtree(child_id)

    for node_id in order:
        if regex.search(paths.get(node_id, "")):
            emit_subtree(node_id)

    return ", ".join(labels)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
