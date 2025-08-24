"""
Copyright (c) 2025 Ma Zhaojia

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from .relaxengine import Scheduler, Worker
from .baseline import ensure_directory, run_baseline
from .utils import count_atoms_cif
from .pbc_graph import radius_graph_pbc_cuda

try:
    from . import extensions
    _extensions_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"Extensions not available: {e}. Falling back to PyTorch implementations.")
    extensions = None
    _extensions_available = False

__all__ = [
    "Scheduler",
    "ensure_directory", 
    "run_baseline",
    "count_atoms_cif",
    "Worker",
    "extensions",
    "radius_graph_pbc_cuda",
]