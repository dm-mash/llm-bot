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

from llm_bot.stores import (
    AgentConfig,
    AgentStore,
    ModelConfig,
    ModelStore,
    ProfileConfig,
    ProfileStore,
)


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


class YamlProfileStore:
    """Loads :class:`ProfileConfig` entries from a ``profiles.yaml`` file.

    Mirrors :class:`YamlAgentStore`: the file has a top-level ``profiles`` key
    mapping profile name -> settings. Profiles are orchestration config (style,
    format, constraints) — see :mod:`llm_bot.profiles`.
    """

    def __init__(self, path: str = "data/profiles.yaml") -> None:
        self._file = _YamlFile(path, "profiles")

    def get(self, name: str) -> ProfileConfig:
        return ProfileConfig.from_dict(name, self._file.item(name))

    def list(self) -> list[str]:
        return self._file.names()


class YamlInvariantStore:
    """Loads global invariant entries from an ``invariants.yaml`` file.

    The file has an optional top-level ``kind_labels`` mapping (free-form
    category -> human-readable label) and a top-level ``invariants`` key
    mapping invariant id -> settings (see :mod:`llm_bot.invariants`). Raw
    dicts are returned — parsing/validation belongs to
    :meth:`llm_bot.invariants.Invariant.from_dict`, keeping this store as thin
    as :class:`YamlProfileStore`.
    """

    def __init__(self, path: str = "data/invariants.yaml") -> None:
        self._file = _YamlFile(path, "invariants")
        self._labels: dict[str, str] | None = None

    def _load_labels(self) -> dict[str, str]:
        if self._labels is None:
            if not os.path.exists(self._file.path):
                raise FileNotFoundError(
                    f"Config file '{self._file.path}' not found. Create it from "
                    "invariants.example.yaml."
                )
            with open(self._file.path, "r", encoding="utf-8") as fh:
                document = yaml.safe_load(fh) or {}
            raw = document.get("kind_labels")
            self._labels = (
                {str(k): str(v) for k, v in raw.items()}
                if isinstance(raw, dict)
                else {}
            )
        return self._labels

    def kind_labels(self) -> dict[str, str]:
        """Return the optional ``kind -> label`` mapping (empty when absent)."""
        return dict(self._load_labels())

    def get(self, name: str) -> dict[str, Any]:
        entry = dict(self._file.item(name))
        entry.setdefault("id", name)
        return entry

    def list(self) -> list[str]:
        return self._file.names()