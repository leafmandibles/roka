# Roka SceneGraph nodes copied from comfyui.datamelder
# Source: ../comfyui.datamelder/__init__.py

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


class RK_SceneGraph:
    CATEGORY = "roka/json"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_json": ("STRING", {"multiline": True}),
                "bbox_order": (["ideogram_yxyx", "raw_xyxy"], {"default": "ideogram_yxyx"}),
            },
            "optional": {
                "res_x": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "res_y": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_image", "summary")
    FUNCTION = "preview"

    def preview(self, bbox_json, bbox_order="ideogram_yxyx", res_x=0, res_y=0):
        import math
        import textwrap
        import numpy as np
        import torch
        from PIL import Image, ImageDraw, ImageFont

        data = _rk_json_load(bbox_json, [])
        items = data if isinstance(data, list) else [data]
        items = [item for item in items if isinstance(item, dict) and isinstance(item.get("bbox"), list) and len(item["bbox"]) == 4]

        def to_xyxy(bbox):
            if bbox_order == "ideogram_yxyx":
                y1, x1, y2, x2 = bbox
            else:
                x1, y1, x2, y2 = bbox
            return [float(x1), float(y1), float(x2), float(y2)]

        boxes = [to_xyxy(item["bbox"]) for item in items]
        if not boxes:
            image = Image.new("RGB", (1024, 1024), "#FFFFFF")
            arr = np.array(image).astype(np.float32) / 255.0
            return (torch.from_numpy(arr).unsqueeze(0), "No bbox items found")

        max_coord = max(max(abs(v) for v in box) for box in boxes)
        looks_normalized = bbox_order == "ideogram_yxyx" and max_coord <= 1000

        if res_x and res_y:
            width, height = int(res_x), int(res_y)
        elif looks_normalized:
            width, height = 1024, 1024
        else:
            min_x = min(box[0] for box in boxes)
            min_y = min(box[1] for box in boxes)
            max_x = max(box[2] for box in boxes)
            max_y = max(box[3] for box in boxes)
            content_w = max(1.0, max_x - min_x)
            content_h = max(1.0, max_y - min_y)
            aspect = content_w / content_h
            target_area = 1024 * 1024
            width = max(content_w, math.sqrt(target_area * aspect))
            height = max(content_h, width / aspect)
            width = int(math.ceil(width / 64.0) * 64)
            height = int(math.ceil(height / 64.0) * 64)

        image = Image.new("RGB", (width, height), "#FFFFFF")
        draw = ImageDraw.Draw(image)

        if looks_normalized:
            scaled_boxes = [
                [
                    box[0] / 1000.0 * width,
                    box[1] / 1000.0 * height,
                    box[2] / 1000.0 * width,
                    box[3] / 1000.0 * height,
                ]
                for box in boxes
            ]
        else:
            min_x = min(box[0] for box in boxes)
            min_y = min(box[1] for box in boxes)
            max_x = max(box[2] for box in boxes)
            max_y = max(box[3] for box in boxes)
            content_w = max_x - min_x
            content_h = max_y - min_y
            offset_x = (width - content_w) / 2.0 - min_x
            offset_y = (height - content_h) / 2.0 - min_y
            scaled_boxes = [[box[0] + offset_x, box[1] + offset_y, box[2] + offset_x, box[3] + offset_y] for box in boxes]

        palette = ["#BFE3FF", "#FFD1DC", "#D6F5D6", "#FFF0B8", "#E4D7FF", "#FFDCC2", "#C8F4F9", "#F3C7E8"]
        outline_palette = ["#2F80C9", "#C74368", "#4C9A4C", "#C09A18", "#7A5BC7", "#C46A2B", "#2A9AA8", "#B34B95"]
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for idx, (item, box) in enumerate(zip(items, scaled_boxes), start=1):
            fill_color = palette[(idx - 1) % len(palette)]
            outline_color = outline_palette[(idx - 1) % len(outline_palette)]
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=max(2, width // 512))
            label = f"{idx}: {item.get('desc', '')}".strip()
            label = textwrap.shorten(label, width=80, placeholder="…")
            tx, ty = x1 + 6, max(0, y1 + 6)
            draw.rectangle([tx - 3, ty - 3, min(width, tx + 420), ty + 17], fill="#FFFFFF")
            draw.text((tx, ty), label, fill=outline_color, font=font)

        arr = np.array(image).astype(np.float32) / 255.0
        summary = f"Scene graph preview: {len(items)} boxes on {width}x{height}; order={bbox_order}; normalized={looks_normalized}"
        return (torch.from_numpy(arr).unsqueeze(0), summary)



# ─────────────────────────────────────────────────────────────────
#  SAM3 multi-label scene graph helpers
# ─────────────────────────────────────────────────────────────────


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
    # Always use roka's own SAM3Grounding. Scanning sys.modules for any class
    # named "SAM3Grounding" picks up other SAM3 node packs (e.g. comfyui-sam3),
    # whose class has an incompatible API and breaks segment().
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


class RK_LoadSAM3Model:
    CATEGORY = "roka/sam3"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "models/sam3/sam3.pt",
                    "tooltip": "Path to SAM3 model checkpoint. No download logic; file must already exist."
                }),
            }
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"

    def load_model(self, model_path):
        LoadSAM3Model = _rk_import_sam3_loader()
        return LoadSAM3Model().load_model(model_path, "")


class RK_SpacyFilter:
    CATEGORY = "roka/text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["nominal_nouns", "tokens", "noun_chunks", "noun_chunk_heads"], {"default": "nominal_nouns"}),
                "pos_filter": ("STRING", {"default": "NOUN,PROPN"}),
                "exclude": ("STRING", {"default": "illustration,camera,photograph,image,photo,picture,quality"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("list", "enricher_json")
    FUNCTION = "filter"

    def filter(self, text, mode="nominal_nouns", pos_filter="NOUN,PROPN", exclude="illustration,camera,photograph,image,photo,picture,quality"):
        import re

        raw = text or ""
        allowed_pos = {p.strip().upper() for p in (pos_filter or "NOUN,PROPN").split(",") if p.strip()}
        excluded = {p.strip().lower() for p in (exclude or "").split(",") if p.strip()}

        def add_unique(items):
            out, seen = [], set()
            for item in items:
                cleaned = re.sub(r"\s+", " ", item.strip().lower())
                if not cleaned or cleaned in excluded or cleaned in seen:
                    continue
                seen.add(cleaned)
                out.append(cleaned)
            return out

        try:
            import spacy
            if not hasattr(RK_SpacyFilter, "_nlp"):
                RK_SpacyFilter._nlp = spacy.load("en_core_web_sm")
            doc = RK_SpacyFilter._nlp(raw)

            enricher = {}
            for chunk in doc.noun_chunks:
                if chunk.root.pos_ in allowed_pos:
                    head = re.sub(r"\s+", " ", chunk.root.lemma_.strip().lower())
                    phrase = re.sub(r"\s+", " ", chunk.text.strip().lower())
                    if head and phrase and phrase not in excluded:
                        enricher.setdefault(head, [])
                        if phrase not in enricher[head]:
                            enricher[head].append(phrase)

            if mode == "tokens":
                candidates = [t.lemma_ for t in doc if t.pos_ in allowed_pos and not t.is_stop and not t.is_punct]
            elif mode == "noun_chunks":
                candidates = []
                for chunk in doc.noun_chunks:
                    if chunk.root.pos_ in allowed_pos:
                        candidates.append(chunk.text)
            elif mode == "noun_chunk_heads":
                candidates = []
                for chunk in doc.noun_chunks:
                    if chunk.root.pos_ in allowed_pos:
                        candidates.append(chunk.root.lemma_)
            else:  # nominal_nouns: noun chunk heads plus standalone nouns not covered by chunks
                candidates = []
                covered = set()
                for chunk in doc.noun_chunks:
                    if chunk.root.pos_ in allowed_pos:
                        candidates.append(chunk.root.lemma_)
                        covered.update(t.i for t in chunk)
                for token in doc:
                    if token.i not in covered and token.pos_ in allowed_pos and not token.is_stop and not token.is_punct:
                        candidates.append(token.lemma_)
            return (", ".join(add_unique(candidates)), _rk_json_dump(enricher))
        except Exception:
            # Fallback for environments without spaCy/model: comma-list cleanup + rough noun-ish last word.
            chunks = [p.strip() for p in re.split(r"[,\n]+", raw) if p.strip()] if ("," in raw or "\n" in raw) else re.findall(r"[A-Za-z][A-Za-z'-]*", raw)
            stop = {"a", "an", "the", "with", "and", "or", "in", "on", "at", "of", "to", "her", "his", "their", "is", "are"}
            candidates = []
            for chunk in chunks:
                words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]*", chunk) if w.lower() not in stop]
                if words:
                    candidates.append(words[-1])
            out = add_unique(candidates)
            return (", ".join(out), _rk_json_dump({item: [item] for item in out}))



class RK_SAM3TextSegmentation:
    CATEGORY = "roka/sam3"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL",),
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {"default": "person, clothing, hand", "multiline": False}),
                "confidence_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_matches": ("INT", {"default": -1, "min": -1, "max": 200, "step": 1}),
            },
            "optional": {
                "offload_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("masks", "visualisation", "boxes", "json")
    FUNCTION = "segment"

    def segment(self, sam3_model, image, text_prompt, confidence_threshold=0.2, max_matches=-1, offload_model=False):
        import json
        import gc
        import torch
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import comfy.model_management

        SAM3Grounding, comfy_image_to_pil, pil_to_comfy_image = _rk_import_sam3_grounding()
        comfy.model_management.load_models_gpu([sam3_model])
        pil_image = comfy_image_to_pil(image)
        img_w, img_h = pil_image.size
        grounding = SAM3Grounding()

        labels = [part.strip() for part in text_prompt.split(",") if part.strip()]
        all_masks = []
        all_boxes = []
        label_ranges = {}
        start = 0

        remaining = max_matches
        for label in labels:
            if max_matches == 0 or remaining == 0:
                label_ranges[label] = [start, start]
                continue
            per_label_limit = remaining if remaining and remaining > 0 else -1
            masks, _vis, boxes_json, _scores = grounding._segment_grounding(
                sam3_model, pil_image, img_w, img_h, confidence_threshold, label, None, None, per_label_limit
            )
            count = 0 if masks is None else int(masks.shape[0])
            if count == 1 and torch.count_nonzero(masks[0] > 0.5).item() == 0:
                count = 0
            boxes = json.loads(boxes_json) if boxes_json else []
            if count:
                all_masks.append(masks[:count])
                all_boxes.extend(boxes[:count] if boxes else [_rk_mask_bbox(masks[i]) for i in range(count)])
            label_ranges[label] = [start, start + count]
            start += count
            if remaining and remaining > 0:
                remaining -= count

        if all_masks:
            flat_masks = torch.cat(all_masks, dim=0)
        else:
            flat_masks = torch.zeros(1, img_h, img_w)

        # pastel labeled MASK visualisation for all categories
        vis = pil_image.convert("RGBA")
        colors = [(80, 170, 255, 120), (80, 220, 150, 120), (255, 190, 80, 120), (210, 150, 255, 120), (255, 130, 170, 120)]
        label_by_index = {}
        for label, (a, b) in label_ranges.items():
            for i in range(a, b):
                label_by_index[i] = label
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for i in range(int(flat_masks.shape[0])):
            if i >= len(all_boxes):
                continue
            color = colors[i % len(colors)]
            mask_np = (flat_masks[i].detach().cpu().numpy() > 0.5).astype(np.uint8) * color[3]
            alpha = Image.fromarray(mask_np, mode="L")
            color_layer = Image.new("RGBA", (img_w, img_h), color[:3] + (0,))
            color_layer.putalpha(alpha)
            vis = Image.alpha_composite(vis, color_layer)

            draw = ImageDraw.Draw(vis, "RGBA")
            x1, y1, x2, y2 = [int(v) for v in all_boxes[i]]
            draw.rectangle([x1, y1, x2, y2], outline=color[:3] + (255,), width=max(2, img_w // 512))
            label = f"{i}: {label_by_index.get(i, 'item')}"
            draw.rectangle([x1 + 3, max(0, y1 + 3), min(img_w, x1 + 220), max(18, y1 + 22)], fill=(255, 255, 255, 210))
            draw.text((x1 + 6, max(0, y1 + 6)), label, fill=(0, 0, 0, 255), font=font)
        vis_tensor = pil_to_comfy_image(vis.convert("RGB"))

        if offload_model:
            sam3_model.unpatch_model()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (flat_masks, vis_tensor, json.dumps(all_boxes, indent=2), json.dumps(label_ranges, indent=2))



class RK_SAM3SceneGraph:
    CATEGORY = "roka/sam3"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json": ("STRING", {"multiline": True}),
                "masks": ("MASK",),
            },
            "optional": {
                "foreground_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("scenegraph", "bboxes")
    FUNCTION = "build"

    def build(self, json, masks, foreground_mask=None):
        import json as jsonlib
        import torch

        label_ranges = _rk_json_load(json, {})
        n = int(masks.shape[0]) if masks is not None else 0

        def mask_2d(mask):
            mask = mask > 0.5
            while mask.dim() > 2:
                mask = mask.any(dim=0)
            return mask

        def infer_bbox_from_mask(idx):
            mi = mask_2d(masks[idx])
            ys, xs = torch.where(mi)
            if xs.numel() == 0 or ys.numel() == 0:
                return [0, 0, 0, 0]
            return [
                int(xs.min().item()),
                int(ys.min().item()),
                int(xs.max().item()) + 1,
                int(ys.max().item()) + 1,
            ]

        boxes = [infer_bbox_from_mask(i) for i in range(n)]

        mask_items = [mask_2d(masks[i]) if masks is not None and i < masks.shape[0] else None for i in range(n)]
        areas = [mask_items[i].sum().item() if mask_items[i] is not None else 0 for i in range(n)]

        def label_for(idx):
            for label, span in label_ranges.items():
                if span[0] <= idx < span[1]:
                    return label
            return "item"

        fg = None
        if foreground_mask is not None:
            fg = mask_2d(foreground_mask)

        nodes = []
        for i in range(n):
            parent = None
            best = 0
            mi = mask_items[i]
            for j in range(n):
                if mi is None or mask_items[j] is None or i == j or areas[j] <= areas[i]:
                    continue
                overlap = torch.logical_and(mi, mask_items[j]).sum().item()
                ratio = overlap / max(1, areas[i])
                if ratio > 0.6 and areas[j] > best:
                    parent = j
                    best = areas[j]
            foreground = None
            if fg is not None and mi is not None:
                overlap = torch.logical_and(mi, fg).sum().item()
                foreground = overlap / max(1, areas[i]) > 0.2
            nodes.append({"id": i, "label": label_for(i), "bbox": boxes[i] if i < len(boxes) else None, "parent_id": parent, "foreground": foreground})
        return (jsonlib.dumps(nodes, indent=2), jsonlib.dumps(boxes, indent=2))



class RK_SceneGraphReducer:
    CATEGORY = "roka/sam3"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenegraph": ("STRING", {"multiline": True}),
                "bboxes": ("STRING", {"multiline": True}),
                "masks": ("MASK",),
            },
            "optional": {
                "foreground_mask": ("MASK",),
                "depth": ("INT", {"default": -1, "min": -1, "max": 100}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("reduced_scenegraph",)
    FUNCTION = "reduce"

    def reduce(self, scenegraph, bboxes, masks, foreground_mask=None, depth=-1):
        import json as jsonlib
        import torch

        nodes = _rk_json_load(scenegraph, [])
        boxes = _rk_json_load(bboxes, [])
        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(boxes, list):
            boxes = []

        def mask_2d(mask):
            mask = mask > 0.5
            while mask.dim() > 2:
                mask = mask.any(dim=0)
            return mask

        def clean_label(value):
            value = str(value or "item").strip()
            return value or "item"

        node_by_id = {}
        ordered_ids = []
        for fallback, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            try:
                nid = int(node.get("id", fallback))
            except Exception:
                nid = fallback
            node_by_id[nid] = node
            ordered_ids.append(nid)

        n_masks = int(masks.shape[0]) if masks is not None else 0
        if not ordered_ids and n_masks:
            for idx in range(n_masks):
                node_by_id[idx] = {"id": idx, "label": "item", "parent_id": None}
                ordered_ids.append(idx)

        def bbox_for(nid):
            node = node_by_id.get(nid, {})
            box = boxes[nid] if isinstance(nid, int) and 0 <= nid < len(boxes) else node.get("bbox")
            if isinstance(box, list) and len(box) == 4:
                return [int(v) for v in box]
            if masks is not None and isinstance(nid, int) and 0 <= nid < n_masks:
                return _rk_mask_bbox(mask_2d(masks[nid]))
            return None

        fg = mask_2d(foreground_mask) if foreground_mask is not None else None

        def is_foreground(nid):
            node = node_by_id.get(nid, {})
            value = node.get("foreground")
            if isinstance(value, bool):
                return value
            if fg is not None and masks is not None and 0 <= nid < n_masks:
                mi = mask_2d(masks[nid])
                area = mi.sum().item()
                overlap = torch.logical_and(mi, fg).sum().item()
                return overlap / max(1, area) > 0.2
            return False

        children = {}
        for nid in ordered_ids:
            parent = node_by_id.get(nid, {}).get("parent_id")
            try:
                parent = int(parent) if parent is not None else None
            except Exception:
                parent = None
            if parent in node_by_id and parent != nid:
                children.setdefault(parent, []).append(nid)
        for child_list in children.values():
            child_list.sort()

        buckets = {True: [], False: []}
        for nid in ordered_ids:
            buckets[is_foreground(nid)].append(nid)

        def descendants_in_bucket(nid, bucket_set):
            out = []
            seen = set()
            def walk(current):
                if current in seen or current not in bucket_set:
                    return
                seen.add(current)
                out.append(current)
                for child in children.get(current, []):
                    walk(child)
            walk(nid)
            return out

        def bbox_union(ids):
            valid = [bbox_for(nid) for nid in ids]
            valid = [box for box in valid if isinstance(box, list) and len(box) == 4]
            if not valid:
                return None
            return [min(b[0] for b in valid), min(b[1] for b in valid), max(b[2] for b in valid), max(b[3] for b in valid)]

        def label_union(ids):
            labels, seen = [], set()
            for nid in ids:
                node = node_by_id.get(nid, {})
                label = clean_label(node.get("label") or node.get("caption") or node.get("desc"))
                key = label.lower()
                if key not in seen:
                    seen.add(key)
                    labels.append(label)
            return ", ".join(labels) if labels else "item"

        def grouped_siblings(ids):
            by_label = {}
            for nid in ids:
                label = clean_label(node_by_id.get(nid, {}).get("label"))
                by_label.setdefault(label.lower(), []).append(nid)
            out = []
            consumed = set()
            for nid in ids:
                if nid in consumed:
                    continue
                label = clean_label(node_by_id.get(nid, {}).get("label"))
                group = by_label.get(label.lower(), [nid])
                if len(group) == 2:
                    out.append(group[:])
                    consumed.update(group)
                else:
                    out.append([nid])
                    consumed.add(nid)
            return out

        out = []
        next_id = 0
        target_depth = int(depth) if depth is not None else -1

        def add_node(label, parent_id, foreground, bbox=None, source_ids=None, source_depth=None):
            nonlocal next_id
            rid = next_id
            next_id += 1
            item = {"id": rid, "label": label, "bbox": bbox, "parent_id": parent_id, "foreground": foreground}
            if source_ids is not None:
                item["source_ids"] = source_ids
            if source_depth is not None:
                item["depth"] = source_depth
            out.append(item)
            return rid

        def emit_merge(source_roots, parent_id, foreground_value, bucket_set, level):
            source_ids = []
            for root_id in source_roots:
                source_ids.extend(descendants_in_bucket(root_id, bucket_set))
            source_ids = sorted(dict.fromkeys(source_ids))
            return add_node(label_union(source_ids), parent_id, foreground_value, bbox_union(source_ids), source_ids, level)

        def emit_preserved(nid, parent_id, foreground_value, bucket_set, level):
            if target_depth >= 0 and level >= target_depth:
                return emit_merge([nid], parent_id, foreground_value, bucket_set, level)

            node = node_by_id.get(nid, {})
            new_id = add_node(clean_label(node.get("label") or node.get("caption") or node.get("desc")), parent_id, foreground_value, bbox_for(nid), [nid], level)
            child_ids = [child for child in children.get(nid, []) if child in bucket_set]
            for group in grouped_siblings(child_ids):
                if len(group) == 2:
                    if target_depth < 0 or level + 1 >= target_depth:
                        emit_merge(group, new_id, foreground_value, bucket_set, level + 1)
                    else:
                        # Before the reduction boundary, keep both nodes so tree structure remains explicit.
                        for child in group:
                            emit_preserved(child, new_id, foreground_value, bucket_set, level + 1)
                else:
                    emit_preserved(group[0], new_id, foreground_value, bucket_set, level + 1)
            return new_id

        for foreground_value, group_label in ((True, "foreground"), (False, "background")):
            bucket_ids = buckets[foreground_value]
            group_id = add_node(group_label, None, foreground_value)
            bucket_set = set(bucket_ids)
            roots = []
            for nid in bucket_ids:
                parent = node_by_id.get(nid, {}).get("parent_id")
                try:
                    parent = int(parent) if parent is not None else None
                except Exception:
                    parent = None
                if parent not in bucket_set:
                    roots.append(nid)
            roots.sort()
            for group in grouped_siblings(roots):
                if len(group) == 2 and target_depth == 0:
                    emit_merge(group, group_id, foreground_value, bucket_set, 0)
                else:
                    for root in group:
                        emit_preserved(root, group_id, foreground_value, bucket_set, 0)

        return (jsonlib.dumps(out, indent=2),)

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


class RK_SceneGraphSegments:
    CATEGORY = "roka/sam3"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenegraph": ("STRING", {"multiline": True}),
            },
            "optional": {
                "enricher_json": ("STRING", {"multiline": True, "default": "{}"}),
                "depth": ("INT", {"default": -1, "min": -1, "max": 100}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("segments", "igv4_json", "igv4_json_merged")
    FUNCTION = "build"

    def build(self, scenegraph, enricher_json="{}", depth=-1):
        nodes = _rk_json_load(scenegraph, [])
        enricher = _rk_json_load(enricher_json, {})
        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(enricher, dict):
            enricher = {}

        segments = []
        elements = []
        node_by_id = {}
        children = {}

        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            node_by_id[node_id] = node
            parent_id = node.get("parent_id")
            if parent_id is not None:
                children.setdefault(parent_id, []).append(node_id)

            label = node.get("label") or node.get("noun") or "item"
            caption = _rk_scene_caption(label, enricher)
            bbox = node.get("bbox")
            segment = {
                "id": node_id,
                "label": str(label or "item").strip() or "item",
                "caption": caption,
                "bbox": bbox,
                "parent_id": parent_id,
                "foreground": node.get("foreground"),
            }
            segments.append(segment)
            if isinstance(bbox, list) and len(bbox) == 4:
                elements.append({"type": "obj", "bbox": bbox, "desc": caption})

        segment_by_id = {segment["id"]: segment for segment in segments}

        def graph_depth(node_id):
            node = node_by_id.get(node_id)
            if not isinstance(node, dict):
                return 0
            parent_id = node.get("parent_id")
            seen = {node_id}
            d = 0
            while parent_id is not None and parent_id in node_by_id and parent_id not in seen:
                seen.add(parent_id)
                d += 1
                parent_id = node_by_id[parent_id].get("parent_id")
            return d

        def caption_union(node_id):
            captions = []
            seen_nodes = set()
            seen_captions = set()

            def add_caption(value):
                cleaned = str(value or "").strip()
                if cleaned and cleaned not in seen_captions:
                    seen_captions.add(cleaned)
                    captions.append(cleaned)

            def walk(current_id):
                if current_id in seen_nodes:
                    return
                seen_nodes.add(current_id)
                segment = segment_by_id.get(current_id)
                if segment:
                    add_caption(segment.get("caption"))
                for child_id in children.get(current_id, []):
                    walk(child_id)

            walk(node_id)
            return ", ".join(captions)

        if depth is None or int(depth) < 0:
            merged_elements = elements
        else:
            target_depth = int(depth)
            merged_elements = []
            for segment in segments:
                if graph_depth(segment["id"]) != target_depth:
                    continue
                bbox = segment.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    merged_elements.append({"type": "obj", "bbox": bbox, "desc": caption_union(segment["id"])})

        return (_rk_json_dump(segments), _rk_json_dump(elements), _rk_json_dump(merged_elements))


class RK_SceneGraphComposer:
    CATEGORY = "roka/sam3"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenegraph": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("simple_composition", "horizontal_composition", "vertical_composition")
    FUNCTION = "compose"

    def compose(self, scenegraph):
        import json as jsonlib

        nodes = _rk_json_load(scenegraph, [])
        if not isinstance(nodes, list):
            nodes = []

        source_nodes = [node for node in nodes if isinstance(node, dict)]
        node_by_id = {node.get("id"): node for node in source_nodes}
        children = {}
        for node in source_nodes:
            node_id = node.get("id")
            parent = node.get("parent_id")
            if parent in node_by_id and parent != node_id:
                children.setdefault(parent, []).append(node_id)
        for child_list in children.values():
            child_list.sort()

        def bbox_valid(box):
            return isinstance(box, list) and len(box) == 4

        def bbox_union(boxes):
            valid = [[int(v) for v in box] for box in boxes if bbox_valid(box)]
            if not valid:
                return None
            return [
                min(box[0] for box in valid),
                min(box[1] for box in valid),
                max(box[2] for box in valid),
                max(box[3] for box in valid),
            ]

        def label_union(items, fallback="background"):
            labels, seen = [], set()
            for node in items:
                label = str(node.get("caption") or node.get("desc") or node.get("label") or "").strip()
                if not label or label.lower() in {"foreground", "background"}:
                    continue
                key = label.lower()
                if key not in seen:
                    seen.add(key)
                    labels.append(label)
            return ", ".join(labels) if labels else fallback

        def is_foreground_root(node):
            label = str(node.get("label") or "").strip().lower()
            return node.get("foreground") is True and (node.get("parent_id") is None or label == "foreground")

        def is_background_root(node):
            label = str(node.get("label") or "").strip().lower()
            return node.get("foreground") is False and (node.get("parent_id") is None or label == "background")

        foreground_roots = [node.get("id") for node in source_nodes if is_foreground_root(node)]
        background_roots = [node.get("id") for node in source_nodes if is_background_root(node)]
        if not foreground_roots:
            foreground_roots = [node.get("id") for node in source_nodes if node.get("foreground") is True and node.get("parent_id") is None]
        if not background_roots:
            background_roots = [node.get("id") for node in source_nodes if node.get("foreground") is False and node.get("parent_id") is None]

        background_root_set = set(background_roots)
        background_items = [node for node in source_nodes if node.get("foreground") is False and node.get("id") not in background_root_set]
        all_boxes = [node.get("bbox") for node in source_nodes if bbox_valid(node.get("bbox"))]
        canvas = bbox_union(all_boxes) or [0, 0, 1, 1]
        x1, y1, x2, y2 = canvas
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        def copy_foreground(out, id_map, next_id_ref):
            def add_copy(old_id, parent_id):
                old = node_by_id.get(old_id)
                if not isinstance(old, dict):
                    return None
                new_id = next_id_ref[0]
                next_id_ref[0] += 1
                id_map[old_id] = new_id
                copied = dict(old)
                copied["id"] = new_id
                copied["parent_id"] = parent_id
                out.append(copied)
                for child_id in children.get(old_id, []):
                    child = node_by_id.get(child_id, {})
                    if child.get("foreground") is True:
                        add_copy(child_id, new_id)
                return new_id

            for root_id in sorted(foreground_roots):
                add_copy(root_id, None)

        def build_composition(mode):
            out = []
            id_map = {}
            next_id_ref = [0]
            copy_foreground(out, id_map, next_id_ref)

            bg_id = next_id_ref[0]
            next_id_ref[0] += 1
            out.append({"id": bg_id, "label": "background", "bbox": None, "parent_id": None, "foreground": False})

            def add_bg_node(label, items):
                if not items:
                    return
                nonlocal_bg = next_id_ref[0]
                next_id_ref[0] += 1
                out.append({
                    "id": nonlocal_bg,
                    "label": label,
                    "caption": label_union(items, label),
                    "bbox": bbox_union([node.get("bbox") for node in items]),
                    "parent_id": bg_id,
                    "foreground": False,
                    "source_ids": [node.get("id") for node in items],
                })

            if mode == "simple":
                add_bg_node(label_union(background_items, "background"), background_items)
            elif mode == "horizontal":
                groups = [("upper third", []), ("middle third", []), ("lower third", [])]
                for node in background_items:
                    box = node.get("bbox")
                    if not bbox_valid(box):
                        continue
                    cy = (float(box[1]) + float(box[3])) / 2.0
                    rel = (cy - y1) / float(height)
                    idx = 0 if rel < 1.0 / 3.0 else 1 if rel < 2.0 / 3.0 else 2
                    groups[idx][1].append(node)
                for label, items in groups:
                    add_bg_node(label, items)
            else:
                groups = [("left third", []), ("middle third", []), ("right third", [])]
                for node in background_items:
                    box = node.get("bbox")
                    if not bbox_valid(box):
                        continue
                    cx = (float(box[0]) + float(box[2])) / 2.0
                    rel = (cx - x1) / float(width)
                    idx = 0 if rel < 1.0 / 3.0 else 1 if rel < 2.0 / 3.0 else 2
                    groups[idx][1].append(node)
                for label, items in groups:
                    add_bg_node(label, items)

            return jsonlib.dumps(out, indent=2)

        return (build_composition("simple"), build_composition("horizontal"), build_composition("vertical"))


class RK_SceneOverlay:
    CATEGORY = "roka/sam3"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scenegraph": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay",)
    FUNCTION = "overlay"

    def overlay(self, image, scenegraph):
        import numpy as np
        import torch
        from PIL import Image, ImageDraw, ImageFont

        nodes = _rk_json_load(scenegraph, [])
        if not isinstance(nodes, list):
            nodes = []

        img_arr = image[0].detach().cpu().numpy()
        pil_image = Image.fromarray(np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)).convert("RGB")
        w, h = pil_image.size
        base = pil_image.convert("RGBA")
        draw = ImageDraw.Draw(base, "RGBA")
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        fg_color = (60, 220, 130, 255)
        bg_color = (255, 185, 70, 255)
        unknown_color = (120, 190, 255, 255)

        def valid_bbox(box):
            return isinstance(box, list) and len(box) == 4

        def is_wrapper(node):
            label = str(node.get("label") or "").strip().lower()
            return label in {"foreground", "background"} and node.get("parent_id") is None

        for node in nodes:
            if not isinstance(node, dict) or is_wrapper(node):
                continue
            bbox = node.get("bbox")
            if not valid_bbox(bbox):
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            foreground = node.get("foreground")
            color = fg_color if foreground is True else bg_color if foreground is False else unknown_color
            line_width = max(2, w // 512)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            label = str(node.get("caption") or node.get("desc") or node.get("label") or "item").strip() or "item"
            node_id = node.get("id", "?")
            text = f"{node_id}: {label}"
            tx, ty = x1 + 4, max(0, y1 + 4)
            text_box_w = min(w - tx, max(80, min(420, len(text) * 7 + 12)))
            draw.rectangle([tx - 2, ty - 2, tx + text_box_w, ty + 16], fill=(255, 255, 255, 220))
            draw.text((tx, ty), text, fill=(0, 0, 0, 255), font=font)

        arr = np.array(base.convert("RGB")).astype(np.float32) / 255.0
        return (torch.from_numpy(arr).unsqueeze(0),)


class RK_SceneGraphToIdeogram4Json:
    CATEGORY = "roka/sam3"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenegraph": ("STRING", {"multiline": True}),
            },
            "optional": {
                "mode": (["all", "foreground", "background", "leaves"], {"default": "all"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 64.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("elements_json",)
    FUNCTION = "build"

    def build(self, scenegraph, mode="all", megapixels=1.0):
        import json as jsonlib
        import math

        nodes = _rk_json_load(scenegraph, [])
        if not isinstance(nodes, list):
            nodes = []
        nodes = [node for node in nodes if isinstance(node, dict)]

        node_by_id = {node.get("id"): node for node in nodes}
        children = {}
        for node in nodes:
            node_id = node.get("id")
            parent_id = node.get("parent_id")
            if parent_id in node_by_id and parent_id != node_id:
                children.setdefault(parent_id, []).append(node_id)

        def valid_bbox(box):
            return isinstance(box, list) and len(box) == 4

        def is_wrapper(node):
            label = str(node.get("label") or "").strip().lower()
            return label in {"foreground", "background"} and node.get("parent_id") is None

        all_boxes = [node.get("bbox") for node in nodes if valid_bbox(node.get("bbox"))]
        if not all_boxes:
            return (jsonlib.dumps([], indent=2),)

        scene_x1 = min(float(box[0]) for box in all_boxes)
        scene_y1 = min(float(box[1]) for box in all_boxes)
        scene_x2 = max(float(box[2]) for box in all_boxes)
        scene_y2 = max(float(box[3]) for box in all_boxes)
        scene_w = max(1.0, scene_x2 - scene_x1)
        scene_h = max(1.0, scene_y2 - scene_y1)

        target_area = max(1.0, float(megapixels or 1.0) * 1_000_000.0)
        aspect = scene_w / scene_h
        canvas_w = math.sqrt(target_area * aspect)
        canvas_h = canvas_w / aspect

        def include_node(node):
            if is_wrapper(node) or not valid_bbox(node.get("bbox")):
                return False
            if mode == "foreground":
                return node.get("foreground") is True
            if mode == "background":
                return node.get("foreground") is False
            if mode == "leaves":
                return len(children.get(node.get("id"), [])) == 0
            return True

        def norm_x(value):
            scaled = (float(value) - scene_x1) / scene_w * canvas_w
            return max(0, min(1000, round(scaled / canvas_w * 1000)))

        def norm_y(value):
            scaled = (float(value) - scene_y1) / scene_h * canvas_h
            return max(0, min(1000, round(scaled / canvas_h * 1000)))

        elements = []
        for node in nodes:
            if not include_node(node):
                continue
            x1, y1, x2, y2 = node.get("bbox")
            desc = str(node.get("caption") or node.get("desc") or node.get("label") or "item").strip() or "item"
            elements.append({
                "type": "obj",
                "bbox": [norm_y(y1), norm_x(x1), norm_y(y2), norm_x(x2)],
                "desc": desc,
            })

        return (jsonlib.dumps(elements, indent=2, ensure_ascii=False),)


class RK_Ideogram4JsonPromptComposer:
    CATEGORY = "roka/ideogram"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "elements_json": ("STRING", {"multiline": True}),
            },
            "optional": {
                "background": ("STRING", {"multiline": True, "default": ""}),
                "aesthetics": ("STRING", {
                    "multiline": True,
                    "default": "photorealistic editorial image, composition preserved from the source reference",
                }),
                "lighting": ("STRING", {
                    "multiline": True,
                    "default": "natural cinematic light matching the source composition",
                }),
                "photo": ("STRING", {"multiline": True, "default": "high quality realistic photograph"}),
                "medium": ("STRING", {"multiline": True, "default": "photorealistic digital image"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_prompt",)
    FUNCTION = "compose"

    def compose(
        self,
        elements_json,
        background="",
        aesthetics="photorealistic editorial image, composition preserved from the source reference",
        lighting="natural cinematic light matching the source composition",
        photo="high quality realistic photograph",
        medium="photorealistic digital image",
    ):
        elements = _rk_json_load(elements_json, [])
        if isinstance(elements, dict):
            # Accept either a full prompt or a compositional_deconstruction object for convenience.
            if isinstance(elements.get("compositional_deconstruction"), dict):
                elements = elements["compositional_deconstruction"].get("elements", [])
            else:
                elements = elements.get("elements", [])
        if not isinstance(elements, list):
            elements = []

        prompt = {
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
        return (_rk_json_dump(prompt),)


class RK_SceneGraphAsciiRenderer:
    CATEGORY = "roka/sam3"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scenegraph": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ascii",)
    FUNCTION = "render"

    def render(self, scenegraph):
        nodes = _rk_json_load(scenegraph, [])
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

        return ("SceneGraph\n" + "\n".join(lines) if lines else "SceneGraph\n(empty)",)


class RK_SceneGraphRenderer:
    CATEGORY = "roka/sam3"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "json": ("STRING", {"multiline": True}),
                "scenegraph": ("STRING", {"multiline": True}),
            },
            "optional": {
                "foreground_mask": ("MASK",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("render", "json", "ascii")
    FUNCTION = "render"

    def render(self, masks, json, scenegraph, foreground_mask=None, image=None):
        import numpy as np
        import torch
        from PIL import Image, ImageDraw, ImageFont

        nodes = _rk_json_load(scenegraph, [])
        enricher = _rk_json_load(json, {})
        if not isinstance(enricher, dict):
            enricher = {}
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        n, h, w = masks.shape
        if image is not None:
            img_arr = image[0].detach().cpu().numpy()
            img = Image.fromarray(np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)).convert("RGB")
            if img.size != (w, h):
                img = img.resize((w, h), Image.Resampling.LANCZOS)
        else:
            img = Image.new("RGB", (w, h), "#FFFFFF")
        draw = ImageDraw.Draw(img, "RGBA")
        bg_colors = [(120, 190, 255, 105), (155, 210, 255, 105), (95, 165, 230, 105)]
        fg_colors = [(120, 230, 160, 125), (90, 205, 130, 125), (165, 245, 190, 125)]
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        node_by_id = {node.get("id"): node for node in nodes if isinstance(node, dict)}
        base = img.convert("RGBA")
        ideogram_elements = []

        def build_ascii_tree():
            children = {}
            roots = []
            for idx in range(n):
                node = node_by_id.get(idx, {"id": idx, "label": "item", "parent_id": None})
                parent = node.get("parent_id")
                if parent is None or parent not in node_by_id or parent == idx:
                    roots.append(idx)
                else:
                    children.setdefault(parent, []).append(idx)

            for child_list in children.values():
                child_list.sort()
            roots.sort()

            lines = []
            visited = set()

            def node_label(idx):
                node = node_by_id.get(idx, {"id": idx, "label": "item"})
                label = node.get("caption") or node.get("desc") or node.get("label") or "item"
                return f"{idx}: {label}"

            def walk(idx, prefix="", is_last=True):
                if idx in visited:
                    lines.append(f"{prefix}{'└── ' if is_last else '├── '}{node_label(idx)} ↩")
                    return
                visited.add(idx)
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{node_label(idx)}")
                next_prefix = prefix + ("    " if is_last else "│   ")
                child_list = children.get(idx, [])
                for child_pos, child_id in enumerate(child_list):
                    walk(child_id, next_prefix, child_pos == len(child_list) - 1)

            if not roots and n:
                roots = list(range(n))
            for root_pos, root_id in enumerate(roots):
                walk(root_id, "", root_pos == len(roots) - 1)
            for idx in range(n):
                if idx not in visited:
                    walk(idx, "", True)
            return "SceneGraph\n" + "\n".join(lines) if lines else "SceneGraph\n(empty)"

        def norm(value, max_value):
            return max(0, min(1000, round((float(value) / float(max_value)) * 1000)))

        fg = None
        if foreground_mask is not None:
            fg = foreground_mask[0] if foreground_mask.dim() == 3 else foreground_mask

        for i in range(n):
            node = node_by_id.get(i, {"id": i, "label": "item", "foreground": None})
            foreground = node.get("foreground")
            if foreground is None and fg is not None:
                mi = masks[i] > 0.5
                overlap = torch.logical_and(mi, fg > 0.5).sum().item()
                area = mi.sum().item()
                foreground = overlap / max(1, area) > 0.2
            color = (fg_colors if foreground else bg_colors)[i % 3]
            mask_np = (masks[i].detach().cpu().numpy() > 0.5).astype(np.uint8) * color[3]
            alpha = Image.fromarray(mask_np, mode="L")
            layer = Image.new("RGBA", (w, h), color[:3] + (0,))
            layer.putalpha(alpha)
            base = Image.alpha_composite(base, layer)
            draw = ImageDraw.Draw(base, "RGBA")
            bbox = node.get("bbox") or _rk_mask_bbox(masks[i])
            x1, y1, x2, y2 = [int(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], outline=color[:3] + (255,), width=max(2, w // 512))
            caption = f"{i}: {node.get('label', 'item')} p={node.get('parent_id')} fg={node.get('foreground')}"
            draw.rectangle([x1 + 3, max(0, y1 + 3), min(w, x1 + 340), max(18, y1 + 22)], fill=(255, 255, 255, 210))
            draw.text((x1 + 5, max(0, y1 + 5)), caption, fill=(0, 0, 0, 255), font=font)
            label = node.get("label", "item")
            desc = str(node.get("caption") or node.get("desc") or "").strip()
            if not desc:
                desc = _rk_scene_caption(label, enricher)
            ideogram_elements.append({
                "type": "obj",
                "bbox": [norm(y1, h), norm(x1, w), norm(y2, h), norm(x2, w)],
                "desc": desc,
            })
        arr = np.array(base.convert("RGB")).astype(np.float32) / 255.0
        return (torch.from_numpy(arr).unsqueeze(0), _rk_json_dump(ideogram_elements), build_ascii_tree())




NODE_CLASS_MAPPINGS = {
    "RK_SceneGraph": RK_SceneGraph,
    "RK_LoadSAM3Model": RK_LoadSAM3Model,
    "RK_SpacyFilter": RK_SpacyFilter,
    "RK_SAM3TextSegmentation": RK_SAM3TextSegmentation,
    "RK_SAM3SceneGraph": RK_SAM3SceneGraph,
    "RK_SceneGraphReducer": RK_SceneGraphReducer,
    "RK_SceneGraphSegments": RK_SceneGraphSegments,
    "RK_SceneGraphComposer": RK_SceneGraphComposer,
    "RK_SceneOverlay": RK_SceneOverlay,
    "RK_SceneGraphToIdeogram4Json": RK_SceneGraphToIdeogram4Json,
    "RK_Ideogram4JsonPromptComposer": RK_Ideogram4JsonPromptComposer,
    "RK_SceneGraphAsciiRenderer": RK_SceneGraphAsciiRenderer,
    "RK_SceneGraphRenderer": RK_SceneGraphRenderer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RK_SceneGraph": "RK Scene Graph Preview",
    "RK_LoadSAM3Model": "RK Load SAM3 Model",
    "RK_SpacyFilter": "RK spaCy Filter",
    "RK_SAM3TextSegmentation": "RK SAM3 Multi Text Segmentation",
    "RK_SAM3SceneGraph": "RK SAM3 Scene Graph",
    "RK_SceneGraphReducer": "RK SceneGraphReducer",
    "RK_SceneGraphSegments": "RK Scene Graph Segments",
    "RK_SceneGraphComposer": "RK SceneGraphComposer",
    "RK_SceneOverlay": "RK SceneOverlay",
    "RK_SceneGraphToIdeogram4Json": "RK SceneGraphToIdeogram4Json",
    "RK_Ideogram4JsonPromptComposer": "RK Ideogram4 Json Prompt Composer",
    "RK_SceneGraphAsciiRenderer": "RK SceneGraphAsciiRenderer",
    "RK_SceneGraphRenderer": "RK Scene Graph Renderer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
