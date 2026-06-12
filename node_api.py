"""Small annotation-based facade for classic ComfyUI node classes.

This is intentionally tiny: function signatures describe the node, and this module
compiles them into ComfyUI v1-style classes/dicts.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any as TypingAny, get_args, get_origin


@dataclass(frozen=True)
class InputSpec:
    comfy_type: str
    default: TypingAny = inspect.Parameter.empty
    options: dict[str, TypingAny] = field(default_factory=dict)

    def as_input_type(self):
        opts = dict(self.options)
        if self.default is not inspect.Parameter.empty:
            opts.setdefault("default", self.default)
        return (self.comfy_type, opts) if opts else (self.comfy_type,)


class Float(InputSpec):
    def __init__(self, *, default=inspect.Parameter.empty, min=None, max=None, step=None, tooltip=None):
        opts = {}
        if min is not None:
            opts["min"] = min
        if max is not None:
            opts["max"] = max
        if step is not None:
            opts["step"] = step
        if tooltip is not None:
            opts["tooltip"] = tooltip
        super().__init__("FLOAT", default, opts)


class Int(InputSpec):
    def __init__(self, *, default=inspect.Parameter.empty, min=None, max=None, step=None, tooltip=None):
        opts = {}
        if min is not None:
            opts["min"] = min
        if max is not None:
            opts["max"] = max
        if step is not None:
            opts["step"] = step
        if tooltip is not None:
            opts["tooltip"] = tooltip
        super().__init__("INT", default, opts)


class String(InputSpec):
    def __init__(self, *, default=inspect.Parameter.empty, multiline=None, tooltip=None):
        opts = {}
        if multiline is not None:
            opts["multiline"] = multiline
        if tooltip is not None:
            opts["tooltip"] = tooltip
        super().__init__("STRING", default, opts)


class _AnyType(str):
    def __ne__(self, other):
        return False


Any = _AnyType("*")


_PY_TO_COMFY = {
    str: "STRING",
    int: "INT",
    float: "FLOAT",
    bool: "BOOLEAN",
    Any: Any,
}


def _display_name(node_id: str) -> str:
    out = []
    token = ""
    for ch in node_id.replace("_", " "):
        if token and ch.isupper() and not token[-1].isupper():
            out.append(token)
            token = ch
        else:
            token += ch
    if token:
        out.append(token)
    return " ".join(" ".join(out).split())


def _return_types(annotation: TypingAny) -> tuple[str, ...]:
    if annotation is inspect.Signature.empty or annotation is None:
        return ("STRING",)
    origin = get_origin(annotation)
    if origin is tuple:
        return tuple(_PY_TO_COMFY.get(arg, str(arg)) for arg in get_args(annotation))
    return (_PY_TO_COMFY.get(annotation, str(annotation)),)


def _input_spec(param: inspect.Parameter) -> InputSpec:
    ann = param.annotation
    if isinstance(ann, InputSpec):
        return ann

    comfy_type = _PY_TO_COMFY.get(ann, "STRING")
    default = param.default
    if default is inspect.Parameter.empty:
        default = inspect.Parameter.empty
    return InputSpec(comfy_type, default, {})


def node(path: str, *, returns: tuple[str, ...] | list[str] | None = None, display_name: str | None = None):
    """Decorate a function and expose it as a ComfyUI node class.

    path is shaped like "category/path/RK_NodeName". The category is everything
    before the final slash; the node id is the final segment.
    """

    def decorate(fn):
        pure_path = PurePosixPath(path)
        node_id = pure_path.name
        category = str(pure_path.parent) if str(pure_path.parent) != "." else ""
        sig = inspect.signature(fn)

        required = {}
        for pname, param in sig.parameters.items():
            required[pname] = _input_spec(param).as_input_type()

        return_types = _return_types(sig.return_annotation)
        return_names = tuple(returns) if returns is not None else return_types

        def run(self, **kwargs):
            call_kwargs = {}
            for pname, param in sig.parameters.items():
                if pname in kwargs:
                    call_kwargs[pname] = kwargs[pname]
                    continue
                spec = _input_spec(param)
                if spec.default is not inspect.Parameter.empty:
                    call_kwargs[pname] = spec.default
                elif param.default is not inspect.Parameter.empty:
                    call_kwargs[pname] = param.default
            out = fn(**call_kwargs)
            return out if isinstance(out, tuple) else (out,)

        cls = type(
            node_id,
            (),
            {
                "INPUT_TYPES": classmethod(lambda cls: {"required": required}),
                "RETURN_TYPES": return_types,
                "RETURN_NAMES": return_names,
                "FUNCTION": "run",
                "CATEGORY": category,
                "DESCRIPTION": (fn.__doc__ or "").strip(),
                "run": run,
            },
        )

        NODE_CLASS_MAPPINGS[node_id] = cls
        NODE_DISPLAY_NAME_MAPPINGS[node_id] = display_name or _display_name(node_id)
        return fn

    return decorate


NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
