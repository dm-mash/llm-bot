"""YAML-backed implementations of :class:`ModelStore` and :class:`AgentStore`.

Reads ``models.yaml`` (key ``models``) and ``agents.yaml`` (key ``agents``).
``${VAR}`` placeholders in values are resolved from the environment when the
entry is materialised (see :func:`llm_bot.stores._resolve_env`), so secret
credentials never need to be committed.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

from llm_bot.stores import AgentConfig, AgentStore, ModelConfig, ModelStore


class _YamlFile:
    """Lazily loads a YAML document and exposes a named sub-map."""

    def __init__(self, path: str, key: str) -> None:
        self.path = path
        self.key = key
        self._raw: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._raw is None:
            if not os.path.exists(self.path):
                raise FileNotFoundError(
                    f"Config file '{self.path}' not found. Create it from the "
                    f"corresponding *.example.yaml template."
                )
            with open(self.path, "r", encoding="utf-8") as fh:
                document = yaml.safe_load(fh) or {}
            section = document.get(self.key)
            if section is None:
                raise ValueError(
                    f"Config file '{self.path}' has no top-level '{self.key}' key."
                )
            if not isinstance(section, dict):
                raise ValueError(
                    f"'{self.key}' in '{self.path}' must be a mapping of name -> settings."
                )
            self._raw = section
        return self._raw

    def names(self) -> list[str]:
        return sorted(self._load().keys())

    def item(self, name: str) -> dict[str, Any]:
        data = self._load()
        if name not in data:
            available = ", ".join(sorted(data))
            raise KeyError(f"Unknown {self.key[:-1]} '{name}'. Available: {available}.")
        value = data[name]
        if not isinstance(value, dict):
            raise ValueError(f"'{name}' in '{self.path}' must be a mapping.")
        return value


class YamlModelStore:
    """Loads :class:`ModelConfig` entries from a ``models.yaml`` file."""

    def __init__(self, path: str = "data/models.yaml") -> None:
        self._file = _YamlFile(path, "models")

    def get(self, name: str) -> ModelConfig:
        return ModelConfig.from_dict(name, self._file.item(name))

    def list(self) -> list[str]:
        return self._file.names()


class YamlAgentStore:
    """Loads :class:`AgentConfig` entries from an ``agents.yaml`` file."""

    def __init__(self, path: str = "data/agents.yaml") -> None:
        self._file = _YamlFile(path, "agents")

    def get(self, name: str) -> AgentConfig:
        return AgentConfig.from_dict(name, self._file.item(name))

    def list(self) -> list[str]:
        return self._file.names()