"""
Copyright (c) 2025 Ma Zhaojia

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

CUDA Extension wrapper for vector addition and PBC graph operations.
"""
import torch
from torch.utils.cpp_extension import load
import os

def load_cuda_extension():
    """Load the CUDA extension for vector addition."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot load CUDA extension.")
    
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the CUDA source file
    cuda_file = os.path.join(current_dir, "vector_add.cu")
    
    # Load the extension
    return load(
        name="vector_add_cuda",
        sources=[cuda_file],
        verbose=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
    )

def load_pbc_graph_cuda_extension():
    """Load the CUDA extension for PBC graph operations."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot load CUDA extension.")
    
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the CUDA source file
    cuda_file = os.path.join(current_dir, "pbc_graph.cu")
    
    # Load the extension
    return load(
        name="pbc_graph_cuda",
        sources=[cuda_file],
        verbose=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
    )

# Global variable to store loaded extension
_cuda_extension = None
_pbc_graph_cuda_extension = None

def get_cuda_extension():
    """Get or load the CUDA extension."""
    global _cuda_extension
    if _cuda_extension is None:
        _cuda_extension = load_cuda_extension()
    return _cuda_extension

def get_pbc_graph_cuda_extension():
    """Get or load the PBC graph CUDA extension."""
    global _pbc_graph_cuda_extension
    if _pbc_graph_cuda_extension is None:
        _pbc_graph_cuda_extension = load_pbc_graph_cuda_extension()
    return _pbc_graph_cuda_extension

def vector_add_cuda(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Perform vector addition using CUDA implementation.
    
    Args:
        a: First input tensor (must be on CUDA device)
        b: Second input tensor (must be on CUDA device)
        
    Returns:
        Result tensor of element-wise addition
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    
    if not (a.is_cuda and b.is_cuda):
        raise ValueError("CUDA implementation requires CUDA tensors. Use .cuda() to move tensors to GPU.")
    
    extension = get_cuda_extension()
    return extension.vector_add(a.float(), b.float())
