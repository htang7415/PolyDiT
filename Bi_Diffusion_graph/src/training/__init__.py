"""Training module exports with lazy loading."""

from importlib import import_module

__all__ = [
    "BackboneTrainer",
    "GraphBackboneTrainer",
]


def __getattr__(name):
    if name == "BackboneTrainer":
        return import_module(".trainer_backbone", __name__).BackboneTrainer
    if name == "GraphBackboneTrainer":
        return import_module(".graph_trainer_backbone", __name__).GraphBackboneTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
