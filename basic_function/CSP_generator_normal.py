"""
This module provides the CrystalGenerator class for crystal structure prediction (CSP).

It uses a Sobol sequence-based random search to generate candidate crystal
structures for a given set of molecules and space group, followed by a crude
packing minimization.
"""

# Standard library imports
import itertools
from typing import List, Tuple, Optional, Any

# Third-party imports
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

# Local application/library specific imports
from basic_function import chemical_knowledge
from basic_function import operation
from basic_function import data_classes

# Module-level constants for better readability and maintenance
_VDW_CLASH_FACTOR = 0.9  # Scaling factor for van der Waals radii in collision checks
_SUPERCELL_RANGE = np.arange(-2, 3) # Range for generating supercell translations


class CrystalGenerator:
    """
    Generates candidate crystal structures for Crystal Structure Prediction (CSP).

    The generator takes a list of unique molecules and a space group, then searches
    the conformational space of cell parameters and molecular orientations to
    produce tightly packed, sterically plausible crystal structures.
    """

    def __init__(self,
                 molecules: list[data_classes.Molecule],
                 space_group: int = 1,
                 angles: tuple[float, float] = (45.0, 135.0)):
        """
        Initializes the CrystalGenerator.

        Args:
            molecules: A list of molecule objects (from data_classes) that will form
                       the asymmetric unit.
            space_group: The international space group number (e.g., 1 for P1).
            angles: A tuple (min, max) defining the range for sampling cell angles in degrees.
        """
        if not (0 < space_group <= 230):
            raise ValueError("Space group must be an integer between 1 and 230.")

        self.molecules = molecules
        self.space_group_number = space_group
        self.angle_sampling_range = angles

        # Derived properties from the space group
        self.symmetry_ops = chemical_knowledge.space_group[self.space_group_number][0]
        self.point_group = chemical_knowledge.space_group[self.space_group_number][2]

        # Calculate counts and dimensions
        self.num_asym_molecules = len(self.molecules)
        self.num_total_molecules = len(self.symmetry_ops) * self.num_asym_molecules
        self.atomic_counts_per_molecule = self._calculate_atomic_counts()

        # Determine search space dimensionality
        self.search_dimensions, self.search_dimension_shape = self._determine_search_dimensions()

        # Pre-calculate molecular and crystal properties
        self.max_vdw_radius = self._find_max_vdw_radius()
        self.estimated_packed_volume = self._calculate_estimated_packed_volume()
        self._orient_molecules()

        # Pre-generate supercell translation vectors, sorted by distance from origin
        self.supercell_frac_translations = np.array(
            sorted(list(itertools.product(_SUPERCELL_RANGE, repeat=3)),
                   key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
        )

    def _calculate_atomic_counts(self) -> list[int]:
        """Calculates the number of atoms for each molecule in the asymmetric unit."""
        return [len(mol.atoms) for mol in self.molecules]

    def _orient_molecules(self) -> None:
        """
        Orients each molecule to a standardized principal axis frame.
        This reduces the rotational search space. For details, see: http://sobereva.com/426
        """
        for i, molecule in enumerate(self.molecules):
            if len(molecule.atoms) > 1:
                self.molecules[i] = operation.orient_molecule(molecule)

    def _find_max_vdw_radius(self) -> float:
        """Finds the maximum van der Waals radius among all atoms in all molecules."""
        vdw_max = 0.0
        for molecule in self.molecules:
            elements, _ = molecule.get_ele_and_cart()
            for ele in set(elements):
                vdw_max = max(vdw_max, chemical_knowledge.element_vdw_radii[ele])
        return vdw_max

    def _determine_search_dimensions(self) -> tuple[int, list[int]]:
        """
        Determines the dimensionality of the search space.

        The search space consists of:
        - 3 dimensions for cell angles (alpha, beta, gamma)
        - 3 dimensions for cell lengths (a, b, c)
        - 3 * N dimensions for molecular translations (x, y, z for each of N molecules)
        - 3 * N dimensions for molecular rotations (Euler angles for each of N molecules)

        Returns:
            A tuple containing the total dimension count and a list detailing the
            breakdown of dimensions.
        """
        dim_cell_lengths = 3
        dim_cell_angles = 3
        dim_translations = 3 * self.num_asym_molecules
        dim_rotations = 3 * self.num_asym_molecules
        total_dimension = dim_cell_lengths + dim_cell_angles + dim_translations + dim_rotations
        shape = [dim_cell_lengths, dim_cell_angles, dim_translations, dim_rotations]
        return total_dimension, shape

    def _calculate_estimated_packed_volume(self) -> float:
        """
        Estimates the total volume of all molecules in the unit cell based on their
        van der Waals radii. This is used for heuristics during generation.
        """
        total_volume = 0.0
        for molecule in self.molecules:
            elements, _ = molecule.get_ele_and_cart()
            vdws = np.array([chemical_knowledge.element_vdw_radii[x] for x in elements])
            volumes = (4 / 3) * np.pi * vdws**3
            total_volume += np.sum(volumes)
        return total_volume * len(self.symmetry_ops) # Multiply by Z

    def _map_random_to_angle(self, value: float) -> float:
        """
        Maps a random number from [0, 1] to an angle in the specified range.

        This uses an arcsin distribution to more densely sample angles near the
        midpoint of the range, which can be more efficient if orthogonal angles
        are more likely.
        """
        min_angle, max_angle = self.angle_sampling_range
        angle_range = max_angle - min_angle
        # A non-linear mapping to bias sampling
        a = np.arcsin(2 * value - 1.0) / np.pi
        return (0.5 + a) * angle_range + min_angle

    def _get_cell_angles_from_vector(self, vector: np.ndarray) -> tuple[float, float, float]:
        """
        Determines the three cell angles based on a 3D random vector, respecting
        the constraints of the crystal's point group.
        """
        angle_candidates = [self._map_random_to_angle(v) for v in vector]

        if self.point_group == "Triclinic":
            return angle_candidates[0], angle_candidates[1], angle_candidates[2]
        if self.point_group == "Monoclinic":
            return 90.0, angle_candidates[1], 90.0
        if self.point_group in ["Orthorhombic", "Tetragonal", "Cubic"]:
            return 90.0, 90.0, 90.0
        if self.point_group == "Hexagonal":
            return 90.0, 90.0, 120.0
        if self.point_group == "Trigonal":
            # For rhombohedral lattices described in hexagonal axes, angles are fixed.
            # This assumes a rhombohedral setting where angles are variable and equal.
            return angle_candidates[0], angle_candidates[0], angle_candidates[0]
        # Fallback for safety, though should be covered by above cases
        return 90.0, 90.0, 90.0


    def _get_cell_lengths_from_vector(self,
                                      vector: np.ndarray,
                                      cell_angles: list[float],
                                      rotated_molecules_cart: list[np.ndarray]
                                      ) -> tuple[float, float, float]:
        """
        Determines the three cell lengths based on a 3D random vector and molecule size.

        The method first calculates the minimum bounding box for the rotated molecules,
        then scales the lengths based on the random vector to explore larger volumes.
        """
        # Estimate minimum cell lengths to avoid self-collision within a molecule
        min_lengths = np.zeros(3)
        conversion_matrix = operation.c2f_matrix([[1, 1, 1], cell_angles])
        for cart_coords in rotated_molecules_cart:
            frac_coords = cart_coords @ conversion_matrix
            max_vals = np.max(frac_coords, axis=0)
            min_vals = np.min(frac_coords, axis=0)
            min_lengths = np.maximum(min_lengths, max_vals - min_vals)

        # Add a buffer based on the largest VdW radius
        min_lengths += self.max_vdw_radius * 2

        # Scale the lengths using the random vector to explore the search space
        a = min_lengths[0] + vector[0] * (self.num_total_molecules * min_lengths[0])
        b = min_lengths[1] + vector[1] * (self.num_total_molecules * min_lengths[1])
        c = min_lengths[2] + vector[2] * (self.num_total_molecules * min_lengths[2])

        # Apply constraints based on the point group
        if self.point_group in ["Tetragonal", "Hexagonal"]:
            return a, a, c
        if self.point_group in ["Trigonal", "Cubic"]:
            return a, a, a
        return a, b, c

    def _check_for_collisions(self,
                              atom_elements: np.ndarray,
                              atom_cart_coords: np.ndarray
                              ) -> bool:
        """
        Performs a steric clash test for the generated structure.

        It checks for intermolecular distances that are smaller than the sum of
        the van der Waals radii (with a tolerance factor).

        Args:
            atom_elements: A numpy array of element symbols for all atoms in the supercell.
            atom_cart_coords: A numpy array of Cartesian coordinates for all atoms.

        Returns:
            True if a collision is detected, False otherwise.
        """
        vdw_radii = np.array([chemical_knowledge.element_vdw_radii[el.item()] for el in atom_elements])
        
        start_index = 0
        for i in range(self.num_asym_molecules):
            # Define the asymmetric unit molecule to check against its environment
            num_atoms_in_mol = self.atomic_counts_per_molecule[i]
            end_index = start_index + num_atoms_in_mol
            
            asym_mol_coords = atom_cart_coords[start_index:end_index]
            asym_mol_vdws = vdw_radii[start_index:end_index]
            
            # The rest of the atoms form the environment
            neighbor_coords = atom_cart_coords[end_index:]
            neighbor_vdws = vdw_radii[end_index:]

            # A coarse filter using a bounding box around the asymmetric molecule
            mol_min = np.min(asym_mol_coords, axis=0) - self.max_vdw_radius * 2
            mol_max = np.max(asym_mol_coords, axis=0) + self.max_vdw_radius * 2
            box_indices = np.all((neighbor_coords > mol_min) & (neighbor_coords < mol_max), axis=1)

            if not np.any(box_indices):
                # Move to the next molecule in the asymmetric unit
                num_atoms_in_supercell_mol = num_atoms_in_mol * len(self.supercell_frac_translations) * len(self.symmetry_ops)
                start_index += num_atoms_in_supercell_mol
                continue

            nearby_neighbor_coords = neighbor_coords[box_indices]
            nearby_neighbor_vdws = neighbor_vdws[box_indices]

            # Use KD-Trees for efficient nearest-neighbor search
            tree_asym = cKDTree(asym_mol_coords, compact_nodes=False, balanced_tree=False)
            tree_neighbors = cKDTree(nearby_neighbor_coords, compact_nodes=False, balanced_tree=False)

            # Find all pairs of atoms within the maximum possible interaction distance
            possible_contacts = tree_asym.query_ball_tree(tree_neighbors, self.max_vdw_radius * 2)

            for j, neighbor_indices in enumerate(possible_contacts):
                if not neighbor_indices:
                    continue
                
                # Check precise distances for potential contacts
                diff = asym_mol_coords[j] - nearby_neighbor_coords[neighbor_indices]
                # einsum is a fast way to compute squared norms row-wise
                distances = np.sqrt(np.einsum('ij,ij->i', diff, diff))
                
                sum_radii = (asym_mol_vdws[j] + nearby_neighbor_vdws[neighbor_indices]) * _VDW_CLASH_FACTOR

                if np.any(distances < sum_radii):
                    return True # Collision detected
            
            # Update start index for the next asymmetric molecule
            num_atoms_in_supercell_mol = num_atoms_in_mol * len(self.supercell_frac_translations) * len(self.symmetry_ops)
            start_index += num_atoms_in_supercell_mol

        return False # No collisions found


    def _shrink_cell_dimensions(self, a: float, b: float, c: float, locked_dims: list[bool]
                                ) -> tuple[float, float, float, list[int]]:
        """
        Shrinks the crystal cell along the longest unlocked dimension by 1 Angstrom.
        This is a crude optimization step to pack the molecules more tightly.

        Args:
            a, b, c: Current cell lengths.
            locked_dims: A boolean list [a, b, c] where True means the dimension
                         cannot be shrunk further.

        Returns:
            A tuple of (new_a, new_b, new_c, last_change_indices).
        """
        lengths = [val for val, is_locked in zip([a, b, c], locked_dims) if not is_locked]
        if not lengths:
            return a, b, c, [] # All dimensions are locked

        max_length = max(lengths)
        last_change = []

        # Logic to shrink the largest dimension(s) while respecting point group constraints
        if self.point_group in ["Triclinic", "Monoclinic", "Orthorhombic"]:
            if a == max_length and not locked_dims[0]:
                a -= 1.0
                last_change = [0]
            elif b == max_length and not locked_dims[1]:
                b -= 1.0
                last_change = [1]
            elif c == max_length and not locked_dims[2]:
                c -= 1.0
                last_change = [2]
        elif self.point_group in ["Tetragonal", "Hexagonal"]:
            if (a == max_length or b == max_length) and not locked_dims[0]:
                a -= 1.0
                b -= 1.0
                last_change = [0, 1]
            elif c == max_length and not locked_dims[2]:
                c -= 1.0
                last_change = [2]
        elif self.point_group in ["Trigonal", "Cubic"]:
            if (a == max_length or b == max_length or c == max_length) and not locked_dims[0]:
                a -= 1.0
                b -= 1.0
                c -= 1.0
                last_change = [0, 1, 2]
        
        return a, b, c, last_change

    def _setup_crystal_from_vector(self, vector: np.ndarray
                                   ) -> tuple[Optional[list], Optional[list[np.ndarray]], Optional[list[Any]]]:
        """
        Performs the initial setup of a crystal structure from a random vector.
        This includes setting angles, rotating molecules, and setting initial lengths.
        This helper is used by both `generate` and `_generate_from_vector`.
        """
        # Unpack the Sobol vector into its components for cell parameters and molecules
        # Slicing indices based on the defined search space shape
        s = self.search_dimension_shape
        cell_angle_seed = vector[0:s[1]]
        cell_length_seed = vector[s[1]:s[1]+s[0]]
        move_part_seed = vector[s[1]+s[0] : s[1]+s[0]+s[2]]
        rotate_part_seed = vector[s[1]+s[0]+s[2]:]

        # 1. Set cell angles
        alpha, beta, gamma = self._get_cell_angles_from_vector(cell_angle_seed)
        cell_angles = [alpha, beta, gamma]
        
        # Check for valid cell matrix from angles
        ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
        volume_sqrt_term = 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
        if volume_sqrt_term <= 0:
            print("Failed: Invalid angles cannot form a valid parallelepiped.")
            return None, None, None

        # 2. Rotate molecules
        rotated_molecules_cart = []
        rotated_molecules_ele = []
        rotate_vectors = rotate_part_seed.reshape(-1, 3)
        for r_vec, molecule in zip(rotate_vectors, self.molecules):
            elements, cart_coords = molecule.get_ele_and_cart()
            rotation_matrix = operation.get_rotate_matrix(r_vec)
            rotated_cart = cart_coords @ rotation_matrix
            rotated_molecules_cart.append(rotated_cart)
            rotated_molecules_ele.append(elements)

        # 3. Set initial cell lengths
        a, b, c = self._get_cell_lengths_from_vector(cell_length_seed, cell_angles, rotated_molecules_cart)
        cell_lengths = [a, b, c]
        
        crystal_params = [cell_lengths, cell_angles, move_part_seed, rotated_molecules_cart, rotated_molecules_ele]
        
        return crystal_params, volume_sqrt_term, rotate_part_seed

    def _build_supercell_for_clash_test(self,
                                        cell_params: list,
                                        rotated_molecules_cart: list[np.ndarray],
                                        rotated_molecules_ele: list[list[str]],
                                        move_part_seed: np.ndarray
                                        ) -> tuple[np.ndarray, np.ndarray, list, list]:
        """
        Builds a supercell and returns all atomic elements and coordinates for clash testing.
        This version correctly handles asymmetric units with multiple, different-sized molecules.
        """
        f2c_matrix = operation.f2c_matrix(cell_params)
        c2f_matrix = operation.c2f_matrix(cell_params)
        supercell_cart_translations = self.supercell_frac_translations @ f2c_matrix

        all_asym_frac_coords = []
        all_asym_elements = []
        
        # Use lists to collect 2D blocks of coordinates and elements. This is efficient.
        sc_cart_blocks = []
        sc_ele_blocks = []

        for i, cart_coords in enumerate(rotated_molecules_cart):
            # Apply translation vector to this molecule's fractional coordinates
            trans_vector = move_part_seed[i * 3:(i + 1) * 3]
            frac_coords = cart_coords @ c2f_matrix + trans_vector
            
            all_asym_frac_coords.append(frac_coords)
            all_asym_elements.append(rotated_molecules_ele[i])

            # Apply symmetry operations
            symm_cart_coords = operation.apply_SYMM(frac_coords, self.symmetry_ops) @ f2c_matrix
            symm_elements_list = [rotated_molecules_ele[i]] * len(self.symmetry_ops)

            # Center molecules that were moved across periodic boundaries
            centroid_frac = np.mean(frac_coords, axis=0)
            centroids_all_symm = operation.apply_SYMM(centroid_frac, self.symmetry_ops)
            for j, cent in enumerate(centroids_all_symm):
                move_to_center = (np.mod(cent, 1) - cent) @ f2c_matrix
                symm_cart_coords[j] += move_to_center

            # --- Core Correction Logic ---
            # 1. Create the full block of atoms for the current molecule type by applying all
            #    supercell translations.
            mol_block_cart_temp = []
            for translation_vec in supercell_cart_translations:
                # Adding the translation vector to all symmetry-equivalent molecules
                translated_coords = symm_cart_coords + translation_vec
                # Reshape to a flat (N_atoms * N_symm, 3) 2D array and append
                mol_block_cart_temp.append(translated_coords.reshape(-1, 3))
            
            # 2. Stack all translated blocks for this molecule type into a single 2D array
            sc_cart_blocks.append(np.vstack(mol_block_cart_temp))
            
            # 3. Handle the corresponding elements, ensuring they are flattened correctly
            num_translations = len(self.supercell_frac_translations)
            ele_block = np.array(symm_elements_list * num_translations).reshape(-1, 1)
            sc_ele_blocks.append(ele_block)

        # After iterating through all molecule types, stack their respective complete blocks
        final_sc_cart = np.vstack(sc_cart_blocks)
        final_sc_ele = np.vstack(sc_ele_blocks)

        return final_sc_cart, final_sc_ele, all_asym_frac_coords, all_asym_elements
    
    def _create_final_crystal_object(self,
                                     cell_params: list,
                                     asym_frac_coords: list,
                                     asym_elements: list,
                                     seed: Any
                                     ) -> data_classes.Crystal:
        """Creates the final Crystal object from the successful structure."""
        
        flat_elements = np.concatenate(asym_elements, axis=0).reshape(-1, 1)
        flat_frac_coords = np.concatenate(asym_frac_coords, axis=0).reshape(-1, 3)

        atoms = []
        for ele, frac in zip(flat_elements, flat_frac_coords):
            atoms.append(data_classes.Atom(element=ele.item(), frac_xyz=frac))
        
        return data_classes.Crystal(
            cell_para=cell_params,
            atoms=atoms,
            comment=str(seed),
            system_name=str(seed),
            space_group=self.space_group_number,
            SYMM=self.symmetry_ops
        )

    def generate(self,
                 seed: Any = "unknown",
                 test: bool = False,
                 densely_pack_method: bool = False,
                 frame_tolerance: float = 1.5
                 ) -> Optional[data_classes.Crystal]:
        """
        The main generation method.

        Uses a Sobol sequence to get a random vector, then attempts to build and
        pack a crystal structure through an iterative shrinking process.

        Args:
            seed: A seed for the Sobol sequence generator. If "unknown", an error is raised.
            test: A flag for enabling verbose test-mode output (prints cycle number).
            densely_pack_method: If True, applies a heuristic to shrink very large
                                 initial volumes.
            frame_tolerance: Tolerance for checking if the final structure is a 2D slab.

        Returns:
            A `data_classes.Crystal` object if a valid structure is found, otherwise `None`.
        """
        if seed == "unknown":
            raise ValueError("A seed must be provided for the Sobol generator.")

        sobol_gen = qmc.Sobol(d=self.search_dimensions, seed=seed)
        initial_vector = sobol_gen.random(n=1).flatten()
        
        setup_result, volume_sqrt_term, _ = self._setup_crystal_from_vector(initial_vector)
        if setup_result is None:
            return None # Invalid initial angles
            
        cell_lengths, cell_angles, move_part_seed, rot_carts, rot_eles = setup_result
        a, b, c = cell_lengths
        alpha, beta, gamma = cell_angles
        
        # Heuristic to shrink extremely sparse initial structures
        if densely_pack_method:
            crystal_volume = a * b * c * np.sqrt(volume_sqrt_term)
            if crystal_volume > self.estimated_packed_volume * 20:
                c = self.estimated_packed_volume * 20 / (a * b * np.sqrt(volume_sqrt_term))

        locked_dims = [False, False, False]
        old_a, old_b, old_c = a, b, c

        for cycle_no in range(1001):
            if cycle_no == 1001:
                print(f"Stopping: Max optimization cycles reached. Seed: {seed}")
                return None
            
            if a < 0 or b < 0 or c < 0:
                print(f"BUG: Negative cell dimension. sg={self.space_group_number}, seed={seed}")
                return None
            
            if test:
                print(f"Cycle: {cycle_no}")

            cell_params = [[a, b, c], [alpha, beta, gamma]]
            
            sc_cart, sc_ele, asym_fracs, asym_eles = self._build_supercell_for_clash_test(
                cell_params, rot_carts, rot_eles, move_part_seed
            )
            
            has_collision = self._check_for_collisions(sc_ele, sc_cart)

            if has_collision:
                if cycle_no == 0:
                    print(f"Failed: Initial structure has collisions. Seed: {seed}")
                    return None
                
                # Collision occurred, so revert to last good state and lock the changed dimension
                a, b, c = old_a, old_b, old_c
                for dim_idx in last_change:
                    locked_dims[dim_idx] = True
            else:
                # No collision, this is a valid (though maybe not dense) structure.
                # Check if optimization is finished (all dimensions are locked).
                if cycle_no > 0 and all(locked_dims):
                    final_crystal = self._create_final_crystal_object(cell_params, asym_fracs, asym_eles, seed)
                    
                    # Final check to filter out 2D slab-like structures
                    if not operation.detect_is_frame_vdw_new(final_crystal, tolerance=frame_tolerance):
                        print(f"Failed: Generated structure is a 2D slab. Seed: {seed}")
                        return None
                    
                    print(f"Success: Generated a valid crystal structure. Seed: {seed}")
                    return final_crystal

                # If no collision and not finished, save current state and shrink further
                old_a, old_b, old_c = a, b, c
                a, b, c, last_change = self._shrink_cell_dimensions(a, b, c, locked_dims)
    
    # ==============================================================================
    # Test-related functions, kept for compatibility, marked as internal.
    # ==============================================================================
    
    def _generate_from_vector(self,
                              seed_vector: np.ndarray,
                              frame_tolerance: float = 1.5
                              ) -> Optional[data_classes.Crystal]:
        """
        Generates a single crystal structure directly from a vector, without optimization.
        This is an internal method intended for testing and analysis.
        Original name: generate_by_vector_2.
        
        Args:
            seed_vector: A numpy array of shape (self.search_dimensions,) defining the structure.
            frame_tolerance: Tolerance for checking if the final structure is a 2D slab.

        Returns:
            A `data_classes.Crystal` object if valid, otherwise `None`.
        """
        if not isinstance(seed_vector, np.ndarray):
            raise TypeError("seed_vector must be a numpy array.")
        
        expected_len = self.search_dimensions
        if len(seed_vector) != expected_len:
            raise ValueError(f"Length of seed_vector must be {expected_len}, got {len(seed_vector)}.")

        setup_result, _, _ = self._setup_crystal_from_vector(seed_vector)
        if setup_result is None:
            return None # Invalid initial angles

        cell_lengths, cell_angles, move_part_seed, rot_carts, rot_eles = setup_result
        cell_params = [cell_lengths, cell_angles]

        sc_cart, sc_ele, asym_fracs, asym_eles = self._build_supercell_for_clash_test(
            cell_params, rot_carts, rot_eles, move_part_seed
        )

        if self._check_for_collisions(sc_ele, sc_cart):
            print("Failed: Structure from vector has collisions.")
            return None

        generated_crystal = self._create_final_crystal_object(
            cell_params, asym_fracs, asym_eles, seed="from_vector"
        )
        
        # Optional: Keep the slab check for consistency
        # if not operation.detect_is_frame_vdw_new(generated_crystal, tolerance=frame_tolerance):
        #     print("Failed: Generated structure is a 2D slab.")
        #     return None
            
        return generated_crystal

    def _is_valid_vector(self, seed_vector: np.ndarray) -> bool:
        """
        Checks if a given vector produces a valid, collision-free structure.
        Internal method for testing.
        """
        return self._generate_from_vector(seed_vector) is not None