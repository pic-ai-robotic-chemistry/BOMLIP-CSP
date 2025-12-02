# -*- coding: utf-8 -*-
"""
Provides functions for converting between different representations of a
crystallographic unit cell (cell parameters and lattice vectors) and for
transforming atomic coordinates between fractional and Cartesian systems.
"""

# --- Standard Library Imports ---
from typing import List, Tuple, Union

# --- Third-Party Imports ---
import numpy as np
import numpy.typing as npt

# --- Type Aliases for Clarity ---
NDArrayFloat = npt.NDArray[np.float64]
CellParameters = Tuple[List[float], List[float]]
CellVectors = Union[List[List[float]], NDArrayFloat]
Coordinates = Union[List[float], NDArrayFloat]


def cell_para_to_vect(
    cell_para: CellParameters, check: bool = False
) -> CellVectors:
    """Converts cell parameters to lattice vectors.

    The lattice vector `a` is aligned with the x-axis. The vector `b` lies in
    the xy-plane.

    Args:
        cell_para: A tuple containing [[a, b, c], [alpha, beta, gamma]],
                   where lengths are in Angstroms and angles are in degrees.
        check: If True, asserts the input shape is correct.

    Returns:
        A 3x3 list of lists representing the cell vectors [a, b, c].
    """
    if check:
        shape_check = np.array(cell_para)
        assert shape_check.shape == (2, 3), "Input `cell_para` must have shape (2, 3)."

    lengths = cell_para[0]
    angles_deg = cell_para[1]
    
    a, b, c = lengths
    alpha, beta, gamma = np.deg2rad(angles_deg)

    cos_a, cos_b, cos_g = np.cos([alpha, beta, gamma])
    sin_g = np.sin(gamma)

    # This term is related to the square of the cell volume.
    # It ensures the cell parameters are physically valid.
    volume_term_sq = (
        1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
    )
    
    # Ensure the argument for sqrt is non-negative
    volume_term = np.sqrt(max(0, volume_term_sq))

    cell_vect = np.zeros((3, 3))
    cell_vect[0, 0] = a
    cell_vect[1, 0] = b * cos_g
    cell_vect[1, 1] = b * sin_g
    cell_vect[2, 0] = c * cos_b
    cell_vect[2, 1] = c * (cos_a - cos_b * cos_g) / sin_g
    cell_vect[2, 2] = c * volume_term / sin_g

    return cell_vect.tolist()


def cell_vect_to_para(cell_vect: CellVectors, check: bool = False) -> CellParameters:
    """Converts lattice vectors to cell parameters.

    Args:
        cell_vect: A 3x3 array-like object representing the lattice vectors.
        check: If True, asserts the input shape is correct.

    Returns:
        A tuple containing [[a, b, c], [alpha, beta, gamma]].
    """
    cell_vect_np = np.array(cell_vect)
    if check:
        assert cell_vect_np.shape == (3, 3), "Input `cell_vect` must have shape (3, 3)."

    vec_a, vec_b, vec_c = cell_vect_np
    
    len_a = np.linalg.norm(vec_a)
    len_b = np.linalg.norm(vec_b)
    len_c = np.linalg.norm(vec_c)
    
    lengths = [len_a, len_b, len_c]
    
    # Calculate angles using the dot product formula; handle potential floating point inaccuracies.
    def _calculate_angle(v1, v2, norm1, norm2):
        cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
        # Clip to handle values slightly outside [-1, 1] due to precision issues
        return np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    alpha_rad = _calculate_angle(vec_b, vec_c, len_b, len_c)
    beta_rad = _calculate_angle(vec_a, vec_c, len_a, len_c)
    gamma_rad = _calculate_angle(vec_a, vec_b, len_a, len_b)
    
    angles_deg = np.rad2deg([alpha_rad, beta_rad, gamma_rad]).tolist()
    
    return (lengths, angles_deg)


def atom_frac_to_cart_by_cell_vect(
    atom_frac: Coordinates, cell_vect: CellVectors, check: bool = False
) -> List[float]:
    """Converts fractional coordinates to Cartesian coordinates using cell vectors.

    Args:
        atom_frac: A 3-element list or array of fractional coordinates.
        cell_vect: A 3x3 matrix of lattice vectors.
        check: If True, asserts input shapes are correct.

    Returns:
        A list of 3 Cartesian coordinates.
    """
    atom_frac_np = np.array(atom_frac)
    cell_vect_np = np.array(cell_vect)

    if check:
        assert cell_vect_np.shape == (3, 3), "Input `cell_vect` must have shape (3, 3)."
        assert atom_frac_np.shape == (3,), "Input `atom_frac` must have 3 elements."

    # The transformation is a linear combination of the basis vectors.
    # atom_cart = frac_x * vec_a + frac_y * vec_b + frac_z * vec_c
    # This is equivalent to a dot product: [fx, fy, fz] @ [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]
    atom_cart = np.dot(atom_frac_np, cell_vect_np)
    return atom_cart.tolist()


def atom_frac_to_cart_by_cell_para(
    atom_frac: Coordinates, cell_para: CellParameters, check: bool = False
) -> List[float]:
    """Converts fractional coordinates to Cartesian using cell parameters.

    Args:
        atom_frac: A 3-element list or array of fractional coordinates.
        cell_para: The cell parameters [[a, b, c], [alpha, beta, gamma]].
        check: If True, performs validation checks in underlying functions.

    Returns:
        A list of 3 Cartesian coordinates.
    """
    cell_vect = cell_para_to_vect(cell_para, check=check)
    return atom_frac_to_cart_by_cell_vect(atom_frac, cell_vect, check=check)


def atom_cart_to_frac_by_cell_vect(
    atom_cart: Coordinates, cell_vect: CellVectors, check: bool = False
) -> List[float]:
    """Converts Cartesian coordinates to fractional coordinates using cell vectors.

    Args:
        atom_cart: A 3-element list or array of Cartesian coordinates.
        cell_vect: A 3x3 matrix of lattice vectors.
        check: If True, asserts input shapes are correct.

    Returns:
        A list of 3 fractional coordinates.
    """
    atom_cart_np = np.array(atom_cart)
    cell_vect_np = np.array(cell_vect)

    if check:
        assert cell_vect_np.shape == (3, 3), "Input `cell_vect` must have shape (3, 3)."
        assert atom_cart_np.shape == (3,), "Input `atom_cart` must have 3 elements."

    # The transformation is atom_frac = atom_cart @ inverse(cell_vect)
    inv_cell_vect = np.linalg.inv(cell_vect_np)
    atom_frac = np.dot(atom_cart_np, inv_cell_vect)
    return atom_frac.tolist()


def atom_cart_to_frac_by_cell_para(
    atom_cart: Coordinates, cell_para: CellParameters, check: bool = False
) -> List[float]:
    """Converts Cartesian coordinates to fractional using cell parameters.

    Args:
        atom_cart: A 3-element list or array of Cartesian coordinates.
        cell_para: The cell parameters [[a, b, c], [alpha, beta, gamma]].
        check: If True, performs validation checks in underlying functions.

    Returns:
        A list of 3 fractional coordinates.
    """
    cell_vect = cell_para_to_vect(cell_para, check=check)
    return atom_cart_to_frac_by_cell_vect(atom_cart, cell_vect, check=check)


def calculate_volume(cell_info: Union[CellParameters, CellVectors]) -> float:
    """Calculates the volume of the unit cell.

    Args:
        cell_info: Can be either cell parameters [[a,b,c], [al,be,ga]] or
                   a 3x3 matrix of cell vectors.

    Returns:
        The volume of the cell in cubic Angstroms.

    Raises:
        ValueError: If the shape of `cell_info` is not (2, 3) or (3, 3).
    """
    cell_info_np = np.array(cell_info)

    if cell_info_np.shape == (3, 3):
        # Input is cell vectors, calculate volume using the scalar triple product.
        return float(np.abs(np.dot(cell_info_np[0], np.cross(cell_info_np[1], cell_info_np[2]))))
    
    elif cell_info_np.shape == (2, 3):
        # Input is cell parameters.
        lengths, angles_deg = cell_info_np
        a, b, c = lengths
        alpha, beta, gamma = np.deg2rad(angles_deg)

        cos_a, cos_b, cos_g = np.cos([alpha, beta, gamma])

        # Standard formula for volume from cell parameters
        volume_sq = (
            a**2 * b**2 * c**2 * (1 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g)
        )
        return float(np.sqrt(max(0, volume_sq)))

    else:
        raise ValueError(f"Cannot understand input shape {cell_info_np.shape} for `cell_info`.")