"""sltop: A top-like queue viewer for Slurm."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("sltop")
except importlib.metadata.PackageNotFoundError:  # Running from a source tree.
    __version__ = "unknown"
