from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .plugin import BenchmarkPlugin


_PLUGINS: dict[str, BenchmarkPlugin] = {}
_BUILTINS_LOADED = False


def register_plugin(plugin: "BenchmarkPlugin") -> None:
    if plugin.name in _PLUGINS:
        raise ValueError(f"Benchmark plugin {plugin.name!r} is already registered.")
    _PLUGINS[plugin.name] = plugin


def get_plugin(name: str) -> "BenchmarkPlugin":
    ensure_builtin_plugins()
    try:
        return _PLUGINS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PLUGINS)) or "<none>"
        raise KeyError(f"Benchmark plugin {name!r} not found. Available: {available}") from exc


def available_plugins() -> list[str]:
    ensure_builtin_plugins()
    return sorted(_PLUGINS)


def ensure_builtin_plugins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from .j1bench.plugin import PLUGIN as j1bench
    from .lawbench.plugin import PLUGIN as lawbench
    from .lexeval.plugin import PLUGIN as lexeval

    for plugin in (lexeval, lawbench, j1bench):
        if plugin.name not in _PLUGINS:
            register_plugin(plugin)
    _BUILTINS_LOADED = True
