"""Self-contained JSON-Schema validation for SLAP messages.

Covers exactly the keyword subset the vendored SLAP schemas use — `type`, `const`,
`enum`, `pattern`, `properties`, `required`, `additionalProperties`, `items`,
`minItems`, `minLength`, `maxLength`, `minimum`, `maximum` — so the slap module carries
no jsonschema dependency (mirroring `tools/schema.py`, but with the `const`/`pattern`/
`minItems` keywords the protocol needs and tool schemas do not).

`validate_message` is the public entry: it routes on `operation`, enforces the envelope
invariants (protocol tag + implemented major version) SLAP mandates, and returns a list
of human-readable error strings — empty means valid. It never raises on a bad message.
"""

from __future__ import annotations

import re
from typing import Any

from suiban.slap.protocol import VERSION, load_schema

_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}

# Major version this implementation speaks (SLAP.md §Versioning: reject a message whose
# major version we do not implement).
_MAJOR = VERSION.split(".", 1)[0]


def validate_instance(instance: Any, schema: dict, path: str = "$") -> list[str]:
    """Validate one instance against a (subset) JSON Schema; return error strings."""
    errors: list[str] = []

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: {instance!r} is not the required constant {schema['const']!r}")

    expected = schema.get("type")
    if expected is not None:
        allowed = _TYPE_CHECKS.get(expected, (object,))
        if isinstance(instance, bool) and expected in ("integer", "number"):
            errors.append(f"{path}: expected {expected}, got boolean")
            return errors
        if not isinstance(instance, allowed):
            errors.append(f"{path}: expected {expected}, got {type(instance).__name__}")
            return errors

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")

    if expected == "string" and isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, instance) is None:
            errors.append(f"{path}: {instance!r} does not match pattern {pattern!r}")

    if expected in ("integer", "number") and isinstance(instance, int | float):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")

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
                errors.extend(validate_instance(instance[key], sub, f"{path}.{key}"))

    if expected == "array" and isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems {schema['minItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(instance):
                errors.extend(validate_instance(item, item_schema, f"{path}[{i}]"))

    return errors


def validate_message(message: Any) -> list[str]:
    """Validate a full SLAP message. Returns human-readable errors (empty = valid).

    Enforces the envelope invariants SLAP mandates before schema-matching: a message
    must be an object, carry `protocol == "SLAP"`, a major version this implementation
    speaks, and a known `operation`. Never raises.
    """
    if not isinstance(message, dict):
        return [f"$: message must be a JSON object, got {type(message).__name__}"]

    errors: list[str] = []
    if message.get("protocol") != "SLAP":
        errors.append(f"$.protocol: not a SLAP message (got {message.get('protocol')!r})")

    version = message.get("version")
    if isinstance(version, str) and version.split(".", 1)[0] != _MAJOR:
        errors.append(
            f"$.version: unsupported major version {version!r} (this implementation "
            f"speaks {VERSION})"
        )

    operation = message.get("operation")
    try:
        schema = load_schema(operation) if isinstance(operation, str) else None
    except KeyError:
        schema = None
    if schema is None:
        errors.append(f"$.operation: unknown or missing operation {operation!r}")
        return errors

    errors.extend(validate_instance(message, schema))
    # De-duplicate while preserving order (protocol/version checks can echo schema ones).
    seen: set[str] = set()
    deduped = [e for e in errors if not (e in seen or seen.add(e))]
    return deduped


def is_valid(message: Any) -> bool:
    """True when `message` is a schema-valid SLAP message."""
    return not validate_message(message)
