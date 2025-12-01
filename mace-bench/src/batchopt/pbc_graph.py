"""
Copyright (c) 2025 Ma Zhaojia

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

CUDA-accelerated PBC graph operations for atomic systems.
"""
import torch
from typing import Optional, List
from .pbc_graph_legacy import get_max_neighbors_mask
from .extensions.cuda_ops import get_pbc_graph_cuda_extension

def radius_graph_pbc_cuda(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc=None,
    dtype=torch.float64,
):
    """
    Memory-efficient CUDA-accelerated version of radius_graph_pbc.
    
    This implementation follows the memory-efficient approach with triple loops
    but accelerates the distance computation using CUDA kernels.
    """
    if pbc is None:
        pbc = [True, True, True]
    
    device = data.pos.device
    batch_size = len(data.natoms)
    
    # Handle PBC settings
    if hasattr(data, "pbc"):
        data.pbc = torch.atleast_2d(data.pbc)
        for i in range(3):
            if not torch.any(data.pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data.pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations."
                )

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute atom pair indices 
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction for PBC
    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    # Take the max over all images for uniformity
    max_rep = [int(2*rep_a1.max().item()), int(2*rep_a2.max().item()), int(2*rep_a3.max().item())]

    # Pre-transpose data_cell for efficiency
    data_cell = torch.transpose(data.cell, 1, 2)
    
    # Use CUDA kernel for the triple loop computation
    # try:
    pbc_graph_cuda = get_pbc_graph_cuda_extension()
    
    # Call the CUDA implementation
    valid_pair_indices, unit_cell, atom_distance_sqr = pbc_graph_cuda.pbc_distance_cuda(
        pos1, pos2, data_cell, 
        num_atoms_per_image_sqr, batch_size, max_rep, float(radius), device
    )
    
    # Map back to original index1 and index2
    if len(valid_pair_indices) > 0:
        index1 = index1.index_select(0, valid_pair_indices.long())
        index2 = index2.index_select(0, valid_pair_indices.long())
    else:
        index1 = torch.empty(0, dtype=torch.long, device=device)
        index2 = torch.empty(0, dtype=torch.long, device=device)
        unit_cell = torch.empty(0, 3, dtype=dtype, device=device)
        atom_distance_sqr = torch.empty(0, dtype=dtype, device=device)

    # Sort index1 in ascending order and rearrange other arrays correspondingly
    if len(index1) > 0:
        sort_indices = torch.argsort(index1)
        index1 = index1[sort_indices]
        index2 = index2[sort_indices]
        unit_cell = unit_cell[sort_indices]
        atom_distance_sqr = atom_distance_sqr[sort_indices]

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image