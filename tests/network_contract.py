from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

OPENAPI_PATH = Path("docs/network-openapi.yaml")


class ContractValidationError(AssertionError):
    pass


@lru_cache(maxsize=1)
def load_openapi() -> dict[str, Any]:
    return yaml.safe_load(OPENAPI_PATH.read_text(encoding="utf-8"))


def validate_schema(instance: Any, schema_name: str) -> None:
    document = load_openapi()
    schemas = document["components"]["schemas"]
    _validate(instance, {"$ref": f"#/components/schemas/{schema_name}"}, schemas, "$")


def validate_device_info_request(payload: dict[str, Any]) -> None:
    validate_schema(payload, "DeviceInfoRequest")


def validate_ws_frame(frame: dict[str, Any]) -> None:
    kind = frame.get("kind")
    schema_name = {
        "device.online": "DeviceOnlineFrame",
        "device.event": "DeviceEventFrame",
        "sensor.batch": "SensorBatchFrame",
    }.get(kind)
    if schema_name is None:
        raise ContractValidationError(f"$.kind has unsupported value {kind!r}")

    validate_schema(frame, schema_name)


def _validate(
    instance: Any,
    schema: dict[str, Any],
    schemas: dict[str, Any],
    path: str,
) -> None:
    if "$ref" in schema:
        ref = schema["$ref"]
        prefix = "#/components/schemas/"
        if not ref.startswith(prefix):
            raise ContractValidationError(f"{path} has unsupported ref {ref}")
        _validate(instance, schemas[ref[len(prefix) :]], schemas, path)
        return

    if "anyOf" in schema:
        errors: list[str] = []
        for candidate in schema["anyOf"]:
            try:
                _validate(instance, candidate, schemas, path)
                return
            except ContractValidationError as err:
                errors.append(str(err))
        raise ContractValidationError(
            f"{path} did not match any allowed schema: {'; '.join(errors)}"
        )

    if "const" in schema and instance != schema["const"]:
        raise ContractValidationError(
            f"{path} expected constant {schema['const']!r}, got {instance!r}"
        )

    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(instance, dict):
            raise ContractValidationError(
                f"{path} expected object, got {type(instance).__name__}"
            )

        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for name in required:
            if name not in instance:
                raise ContractValidationError(f"{path}.{name} is required")

        if schema.get("additionalProperties") is False:
            extra = set(instance) - set(properties)
            if extra:
                raise ContractValidationError(
                    f"{path} contains unsupported properties: {sorted(extra)!r}"
                )

        for name, value in instance.items():
            child_schema = properties.get(name)
            if child_schema is not None:
                _validate(value, child_schema, schemas, f"{path}.{name}")
        return

    if schema_type == "array":
        if not isinstance(instance, list):
            raise ContractValidationError(
                f"{path} expected array, got {type(instance).__name__}"
            )

        min_items = schema.get("minItems")
        if min_items is not None and len(instance) < min_items:
            raise ContractValidationError(f"{path} expected at least {min_items} items")
        max_items = schema.get("maxItems")
        if max_items is not None and len(instance) > max_items:
            raise ContractValidationError(f"{path} expected at most {max_items} items")

        item_schema = schema.get("items")
        if item_schema is not None:
            for index, value in enumerate(instance):
                _validate(value, item_schema, schemas, f"{path}[{index}]")
        return

    if schema_type == "string":
        if not isinstance(instance, str):
            raise ContractValidationError(
                f"{path} expected string, got {type(instance).__name__}"
            )
        pattern = schema.get("pattern")
        if pattern is not None and re.fullmatch(pattern, instance) is None:
            raise ContractValidationError(f"{path} did not match pattern {pattern!r}")
        return

    if schema_type == "integer":
        if isinstance(instance, bool) or not isinstance(instance, int):
            raise ContractValidationError(
                f"{path} expected integer, got {type(instance).__name__}"
            )
        minimum = schema.get("minimum")
        if minimum is not None and instance < minimum:
            raise ContractValidationError(
                f"{path} expected minimum {minimum}, got {instance}"
            )
        return

    if schema_type == "number":
        if isinstance(instance, bool) or not isinstance(instance, (int, float)):
            raise ContractValidationError(
                f"{path} expected number, got {type(instance).__name__}"
            )
        minimum = schema.get("minimum")
        if minimum is not None and instance < minimum:
            raise ContractValidationError(
                f"{path} expected minimum {minimum}, got {instance}"
            )
        return

    if schema_type == "boolean":
        if not isinstance(instance, bool):
            raise ContractValidationError(
                f"{path} expected boolean, got {type(instance).__name__}"
            )
        return

    if schema_type == "null":
        if instance is not None:
            raise ContractValidationError(f"{path} expected null, got {instance!r}")
        return
