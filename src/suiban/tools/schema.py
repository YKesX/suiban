"""Minimal JSON-schema validation for tool arguments.

llama-server decodes tool calls grammar-constrained (--jinja + the tool schemas), so
arguments *parse* by construction — but a scripted/mock backend, a repaired call, or a
semantically-wrong value can still violate the schema. This validator covers the subset
our tool schemas actually use (type / properties / required / enum / items /
additionalProperties / min-max) so we do not need a jsonschema dependency.

TODO(v1.1): swap for the `jsonschema` package if tool schemas outgrow this subset —
kept minimal on purpose while every schema in tools/ is hand-written and simple.
"""

from __future__ import annotations

from typing import Any

_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


def validate(instance: Any, schema: dict, path: str = "$") -> list[str]:
    """Return a list of human-readable error strings; empty list means valid."""
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        allowed = _TYPE_CHECKS.get(expected, (object,))
        # bool is a subclass of int; don't let True pass as an integer/number.
        if isinstance(instance, bool) and expected in ("integer", "number"):
            errors.append(f"{path}: expected {expected}, got boolean")
            return errors
        if not isinstance(instance, allowed):
            errors.append(f"{path}: expected {expected}, got {type(instance).__name__}")
            return errors

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")

    if expected == "object" and isinstance(instance, dict):
        props: dict = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in props:
                    errors.append(f"{path}: unexpected property {key!r}")
        for key, sub in props.items():
            if key in instance:
                errors.extend(validate(instance[key], sub, f"{path}.{key}"))

    if expected == "array" and isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            errors.extend(validate(item, schema["items"], f"{path}[{i}]"))

    if expected in ("integer", "number") and isinstance(instance, int | float):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")

    if expected == "string" and isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")

    return errors
