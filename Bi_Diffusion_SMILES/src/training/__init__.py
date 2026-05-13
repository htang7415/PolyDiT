"""Training module exports.

Keep imports lazy so importing a specific trainer submodule does not pull in
other trainer dependencies at package-import time.
"""

from importlib import import_module

__all__ = ["BackboneTrainer"]


def __getattr__(name):
    if name == "BackboneTrainer":
        return import_module(".trainer_backbone", __name__).BackboneTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
