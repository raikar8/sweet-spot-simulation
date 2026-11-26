"""Rendering algorithms: the reference set plus anything found in user_renderers/."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .base import REGISTRY, Drive, RenderContext, Renderer, register
from . import reference  # noqa: F401  (registers the reference renderers)

USER_DIR = Path(__file__).resolve().parents[2] / "user_renderers"


def load_user_renderers(folder: Path = USER_DIR) -> list[str]:
    """Import every .py file in user_renderers/ so its @register classes appear. Returns errors."""
    errors = []
    if not folder.exists():
        return errors
    for f in sorted(folder.glob("*.py")):
        if f.name.startswith("_"):
            continue
        name = f"user_renderers.{f.stem}"
        try:
            spec = importlib.util.spec_from_file_location(name, f)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        except Exception as e:  # show the problem in the app instead of crashing
            errors.append(f"{f.name}: {type(e).__name__}: {e}")
    return errors


def get(name: str) -> Renderer:
    return REGISTRY[name]()


__all__ = ["REGISTRY", "Drive", "RenderContext", "Renderer", "register", "load_user_renderers", "get"]
