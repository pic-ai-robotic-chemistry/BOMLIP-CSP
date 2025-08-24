# -*- coding: utf-8 -*-
"""
A collection of functions for performing crystallographic and molecular operations,
such as symmetry application, supercell generation, and geometric analysis.
"""

# --- Standard Library Imports ---
import copy
import fractions
import re
from typing import Any, Dict, List, Optional, Tuple, Union

# --- Third-Party Imports ---
import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree as KDTree

# --- Local Application Imports ---
from basic_function import chemical_knowledge, data_classes

# Type aliases for clarity
NDArrayFloat = npt.NDArray[np.float64]
CellVectors = List[List[float]]
SymmetryOperations = List[str]


def is_number(s: str) -> bool:
    """Checks if a string can be interpreted as a number (float or fraction).

    Args:
        s: The input string.

    Returns:
        True if the string represents a number, False otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        # Check for fractional representations like "1/2"
        float(fractions.Fraction(s))
        return True
    except ValueError:
        return False


def _parse_symmetry_operations(
    sym_ops: SymmetryOperations,
) -> Tuple[List[NDArrayFloat], List[NDArrayFloat]]:
    """Parses a list of symmetry operation strings into matrices.

    This is an internal helper function to avoid code duplication in public functions.

    Args:
        sym_ops: A list of symmetry operation strings (e.g., ['x, y, z+1/2']).

    Returns:
        A tuple containing two lists:
        - A list of 3x3 rotation/reflection matrices (M).
        - A list of 1x3 translation vectors (C).

    Raises:
        ValueError: If a symmetry operation string is malformed.
    """
    rotation_matrices = []
    translation_vectors = []

    for sym_op_str in sym_ops:
        sym_op_parts = sym_op_str.lower().replace(" ", "").split(",")
        if len(sym_op_parts) != 3:
            raise ValueError(f"Symmetry operation '{sym_op_str}' is invalid.")

        matrix_m = np.zeros((3, 3))
        matrix_c = np.zeros((1, 3))

        for i, part in enumerate(sym_op_parts):
            # Regex to find elements like '+x', '-y', 'z', '1/2', '-0.5'
            tokens = re.findall(r"([+-]?[xyz0-9./]+)", part)
            for token in tokens:
                token = token.strip()
                if not token:
                    continue

                if "x" in token:
                    matrix_m[0, i] = -1.0 if token.startswith("-") else 1.0
                elif "y" in token:
                    matrix_m[1, i] = -1.0 if token.startswith("-") else 1.0
                elif "z" in token:
                    matrix_m[2, i] = -1.0 if token.startswith("-") else 1.0
                elif is_number(token):
                    matrix_c[0, i] += float(fractions.Fraction(token))
                else:
                    raise ValueError(f"Invalid fragment '{token}' in symmetry operation.")

        rotation_matrices.append(matrix_m)
        translation_vectors.append(matrix_c)

    return rotation_matrices, translation_vectors


def space_group_transfer_for_single_atom(
    frac_xyz: List[float], space_group_ops: SymmetryOperations
) -> List[List[float]]:
    """Applies space group symmetry operations to a single atomic coordinate.

    Args:
        frac_xyz: The fractional coordinates [x, y, z] of a single atom.
        space_group_ops: A list of space group symmetry operation strings.

    Returns:
        A list of all symmetrically equivalent fractional coordinates.
    """
    rot_matrices, trans_vectors = _parse_symmetry_operations(space_group_ops)
    
    equivalent_positions = []
    atom_pos = np.array(frac_xyz)

    for rot, trans in zip(rot_matrices, trans_vectors):
        new_pos = np.dot(atom_pos, rot.T) + trans.squeeze()
        equivalent_positions.append(new_pos.tolist())

    return equivalent_positions


def super_cell(
    crystal: "data_classes.Crystal",
    cell_range: Optional[List[List[int]]] = None,
) -> "data_classes.Crystal":
    """Constructs a supercell from a unit cell.

    Args:
        crystal: The input Crystal object.
        cell_range: A list of ranges for each lattice vector, e.g.,
                    [[-1, 1], [-1, 1], [-1, 1]] creates a 3x3x3 supercell.
                    If None, defaults to [[-1, 1], [-1, 1], [-1, 1]].

    Returns:
        A new Crystal object representing the supercell.
    """
    if cell_range is None:
        cell_range = [[-1, 1], [-1, 1], [-1, 1]]

    dims = [r[1] - r[0] + 1 for r in cell_range]
    
    new_lattice = [
        [dim * val for val in crystal.cell_vect[i]] for i, dim in enumerate(dims)
    ]

    translation_vectors = []
    for h in range(cell_range[0][0], cell_range[0][1] + 1):
        for k in range(cell_range[1][0], cell_range[1][1] + 1):
            for l in range(cell_range[2][0], cell_range[2][1] + 1):
                translation_vectors.append([h, k, l])

    new_atoms = []
    for atom in crystal.atoms:
        for trans_vec in translation_vectors:
            new_frac_xyz = [
                (atom.frac_xyz[i] + trans_vec[i]) / dims[i] for i in range(3)
            ]
            new_atoms.append(
                data_classes.Atom(element=atom.element, frac_xyz=new_frac_xyz)
            )

    if crystal.energy != "unknown":
        total_cells = dims[0] * dims[1] * dims[2]
        new_energy = crystal.energy * total_cells
    else:
        new_energy = "unknown"

    return data_classes.Crystal(
        cell_vect=new_lattice, energy=new_energy, atoms=new_atoms
    )


def orient_molecule(molecule: "data_classes.Molecule") -> "data_classes.Molecule":
    """Orients a molecule along its principal axes of inertia.

    The method uses the Moment of Inertia tensor to define a canonical orientation.
    The molecule's coordinates are modified in-place. For more details, see:
    http://sobereva.com/426

    Args:
        molecule: The Molecule object to be oriented.

    Returns:
        The same Molecule object with its atoms reoriented.
    """
    all_ele, all_cart = molecule.get_ele_and_cart()

    if len(all_cart) <= 1:
        return molecule  # No orientation needed for single atoms or empty molecules.

    masses = np.array([chemical_knowledge.element_masses[el] for el in all_ele])
    relative_position = all_cart - molecule.get_center_of_mass()

    # Calculate the moment of inertia tensor
    I_xx = np.sum(masses * (relative_position[:, 1] ** 2 + relative_position[:, 2] ** 2))
    I_yy = np.sum(masses * (relative_position[:, 0] ** 2 + relative_position[:, 2] ** 2))
    I_zz = np.sum(masses * (relative_position[:, 0] ** 2 + relative_position[:, 1] ** 2))
    I_xy = -np.sum(masses * relative_position[:, 0] * relative_position[:, 1])
    I_xz = -np.sum(masses * relative_position[:, 0] * relative_position[:, 2])
    I_yz = -np.sum(masses * relative_position[:, 1] * relative_position[:, 2])

    I_matrix = np.array([[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]])

    # Eigenvectors of the inertia tensor are the principal axes.
    # np.linalg.eigh is used for symmetric matrices.
    eigenvalues, eigenvectors = np.linalg.eigh(I_matrix)
    principal_axes = eigenvectors.T
    
    # Project the relative positions onto the new axes system.
    new_positions = np.dot(relative_position, principal_axes.T)
    
    molecule.put_ele_cart_back(all_ele, new_positions)
    return molecule


def get_rotate_matrix(v: NDArrayFloat) -> NDArrayFloat:
    """Generates a 3x3 rotation matrix from a 3D vector `v`.

    This function uses a mapping from a 3D vector to a quaternion, which is then
    used to construct the rotation matrix. This method avoids gimbal lock. A
    left-handed coordinate system is assumed.

    Args:
        v: A 3-element NumPy array used to generate the quaternion.

    Returns:
        A 3x3 rotation matrix.
    """
    # Ensure v elements are within valid ranges if necessary, though the
    # formulas handle most inputs gracefully.
    v0_sqrt = np.sqrt(max(v[0], 0))
    v0_1_sqrt = np.sqrt(max(1.0 - v[0], 0))
    
    angle1 = 2.0 * np.pi * v[1]
    angle2 = 2.0 * np.pi * v[2]

    # Quaternion components (x, y, z, w)
    qx = v0_1_sqrt * np.sin(angle1)
    qy = v0_1_sqrt * np.cos(angle1)
    qz = v0_sqrt * np.sin(angle2)
    qw = v0_sqrt * np.cos(angle2)

    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy + 2*qw*qz, 2*qx*qz - 2*qw*qy],
        [2*qx*qy - 2*qw*qz, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz + 2*qw*qx],
        [2*qx*qz + 2*qw*qy, 2*qy*qz - 2*qw*qx, 1 - 2*qx**2 - 2*qy**2]
    ])


def f2c_matrix(
    cell_params: Tuple[List[float], List[float]]
) -> Optional[NDArrayFloat]:
    """Calculates the fractional-to-Cartesian transformation matrix.

    Args:
        cell_params: A tuple containing [[a, b, c], [alpha, beta, gamma]],
                     where lengths are in Angstroms and angles are in degrees.

    Returns:
        The 3x3 transformation matrix, or None if cell parameters are invalid.
    """
    lengths, angles = cell_params
    a, b, c = lengths
    alpha, beta, gamma = np.deg2rad(angles)

    cos_a, cos_b, cos_g = np.cos([alpha, beta, gamma])
    sin_g = np.sin(gamma)

    # Volume calculation term
    volume_term_sq = (
        1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
    )
    if volume_term_sq < 0:
        return None
    
    volume = a * b * c * np.sqrt(volume_term_sq)

    matrix = np.zeros((3, 3))
    matrix[0, 0] = a
    matrix[0, 1] = b * cos_g
    matrix[0, 2] = c * cos_b
    matrix[1, 1] = b * sin_g
    matrix[1, 2] = c * (cos_a - cos_b * cos_g) / sin_g
    matrix[2, 2] = volume / (a * b * sin_g)
    
    return matrix.T


def c2f_matrix(
    cell_params: Tuple[List[float], List[float]]
) -> Optional[NDArrayFloat]:
    """Calculates the Cartesian-to-fractional transformation matrix.

    This is the inverse of the matrix generated by `f2c_matrix`.

    Args:
        cell_params: A tuple containing [[a, b, c], [alpha, beta, gamma]],
                     where lengths are in Angstroms and angles are in degrees.

    Returns:
        The 3x3 transformation matrix, or None if cell parameters are invalid.
    """
    f2c = f2c_matrix(cell_params)
    if f2c is None:
        return None
    
    try:
        return np.linalg.inv(f2c)
    except np.linalg.LinAlgError:
        return None


def apply_SYMM(
    frac_xyz: NDArrayFloat, symm_ops: SymmetryOperations
) -> NDArrayFloat:
    """Applies symmetry operations to a single set of fractional coordinates.

    Args:
        frac_xyz: A NumPy array of fractional coordinates [x, y, z].
        symm_ops: A list of symmetry operation strings.

    Returns:
        A NumPy array of all symmetrically equivalent fractional coordinates.
    """
    rot_matrices, trans_vectors = _parse_symmetry_operations(symm_ops)

    equivalent_positions = [
        np.dot(frac_xyz, rot.T) + trans.squeeze()
        for rot, trans in zip(rot_matrices, trans_vectors)
    ]
    
    return np.array(equivalent_positions)


def apply_SYMM_with_element(
    elements: Union[str, List[str]],
    frac_xyzs: NDArrayFloat,
    symm_ops: SymmetryOperations,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Applies symmetry operations, returning new elements and coordinates.

    Args:
        elements: The element symbol(s) corresponding to the coordinates.
        frac_xyzs: A NumPy array of fractional coordinates.
        symm_ops: A list of symmetry operation strings.

    Returns:
        A tuple containing:
        - A NumPy array of element symbols for each new position.
        - A NumPy array of all symmetrically equivalent fractional coordinates.
    """
    equivalent_positions = apply_SYMM(frac_xyzs, symm_ops)
    num_ops = len(equivalent_positions)
    
    replicated_elements = np.tile(np.array(elements).squeeze(), (num_ops, 1))
    
    return replicated_elements, equivalent_positions


def calculate_longest_diagonal_length(cell_vect: CellVectors) -> float:
    """Calculates the length of the longest space diagonal of a unit cell.

    The longest diagonal connects the origin (0,0,0) to the opposite
    corner (1,1,1) of the unit cell.

    Args:
        cell_vect: The three lattice vectors of the cell.

    Returns:
        The length of the longest diagonal in Angstroms.
    """
    cell_vect_np = np.array(cell_vect)
    diagonal_vector = np.sum(cell_vect_np, axis=0)
    return float(np.linalg.norm(diagonal_vector))


def calculate_distance_of_parallel_plane_in_crystal(cell_vect: CellVectors) -> List[float]:
    """Calculates inter-planar distances for primary crystallographic planes.

    This computes the distances for the (100), (010), and (001) families of planes.

    Args:
        cell_vect: The three lattice vectors [a, b, c] of the cell.

    Returns:
        A list of three distances [d_a, d_b, d_c], where d_a is the distance
        between planes parallel to the b-c plane, and so on.
    """
    distances = []
    vectors = [np.array(v) for v in cell_vect]
    
    # Permutations to calculate distance for each primary plane
    # (a to b-c plane, b to a-c plane, c to a-b plane)
    indices = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]

    for i, j, k in indices:
        point_p = vectors[i]
        plane_v1 = vectors[j]
        plane_v2 = vectors[k]
        
        # Normal vector to the plane defined by plane_v1 and plane_v2
        normal_vector = np.cross(plane_v1, plane_v2)
        
        # Distance from point P to the plane is |N · P| / ||N||
        distance = abs(np.dot(normal_vector, point_p)) / np.linalg.norm(normal_vector)
        distances.append(distance)
        
    return distances


def detect_is_frame_vdw_new(crystal: "data_classes.Crystal", tolerance: float = 1.2) -> bool:
    """Detects if a crystal structure forms a connected framework via VdW radii.

    The method involves:
    1. Expanding the crystal to a P1 symmetry supercell.
    2. Building a 3x3x3 supercell to ensure periodic connections are considered.
    3. Constructing a graph where atoms are nodes and an edge exists if their
       distance is within a scaled sum of their van der Waals radii.
    4. Checking if the largest connected component in the graph is large enough
       to be considered a single, percolating framework.

    Args:
        crystal: The Crystal object to analyze.
        tolerance: A tolerance factor to scale the VdW radii sum.

    Returns:
        True if the structure is a connected framework, False otherwise.
    """
    crystal_temp = copy.deepcopy(crystal)
    crystal_temp.make_p1()
    crystal_temp.move_atom_into_cell()

    # Create a 3x3x3 supercell to check for connectivity across boundaries
    crystal_supercell = super_cell(crystal_temp, cell_range=[[-1, 1], [-1, 1], [-1, 1]])
    
    all_ele, all_carts = crystal_supercell.get_ele_and_cart()

    vdw_radii_map = chemical_knowledge.element_vdw_radii
    vdw_max = max(vdw_radii_map[el] for el in set(all_ele))
    distance_threshold = vdw_max * tolerance * 2

    # KDTree for efficient nearest-neighbor search
    tree = KDTree(all_carts)
    pairs = tree.query_pairs(r=distance_threshold)

    # Build a graph to find connected components
    graph = nx.Graph()
    graph.add_nodes_from(range(len(all_carts)))
    graph.add_edges_from(list(pairs))

    if not graph.nodes:
        return False

    # Find the largest connected component
    largest_cc = max(nx.connected_components(graph), key=len)

    # A heuristic to check for a percolating framework. A connected framework
    # should connect most atoms. The threshold '9' is empirical but robustly
    # distinguishes between isolated molecules and a fully connected lattice.
    # In a 3x3x3 supercell (27 unit cells), a connected framework should involve
    # significantly more atoms than in a few unit cells.
    return len(largest_cc) > 9 * len(crystal_temp.atoms)