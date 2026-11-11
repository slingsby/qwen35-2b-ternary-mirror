"""Pick a ModelAdapter from a Hugging Face config's ``model_type``."""
from __future__ import annotations

from .base import ModelAdapter
from .dense import DenseAdapter
from .qwen3_5 import Qwen35Adapter

_REGISTRY: dict[str, type[ModelAdapter]] = {}


def register(cls: type[ModelAdapter]) -> type[ModelAdapter]:
    for model_type in cls.model_types:
        _REGISTRY[model_type] = cls
    return cls


register(Qwen35Adapter)
register(DenseAdapter)


def registered_adapters() -> list[type[ModelAdapter]]:
    return sorted(set(_REGISTRY.values()), key=lambda c: c.name)


def supported_model_types() -> list[str]:
    return sorted(_REGISTRY)


def adapter_for_config(config, name: str | None = None) -> ModelAdapter:
    if name not in (None, "auto"):
        matches = [c for c in set(_REGISTRY.values()) if c.name == name]
        if not matches:
            raise ValueError(f"unknown adapter {name!r}; known: {sorted({c.name for c in _REGISTRY.values()})}")
        return matches[0](config)
    cls = _REGISTRY.get(config.model_type)
    if cls is None:
        raise ValueError(f"no adapter for model_type {config.model_type!r}; supported: {supported_model_types()}")
    return cls(config)


def adapter_for_model(model) -> ModelAdapter:
    return adapter_for_config(model.config)


def load_adapter(path, *, revision=None, local_files_only=False, name: str | None = None) -> ModelAdapter:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(path, revision=revision, local_files_only=local_files_only,
                                        trust_remote_code=False)
    return adapter_for_config(config, name)
