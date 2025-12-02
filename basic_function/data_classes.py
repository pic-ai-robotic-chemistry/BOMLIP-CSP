"""
This module defines the core data structures for representing atomic structures:
Atom, Crystal, and Molecule.

These classes store information about atomic coordinates, lattice parameters,
and other physical properties, providing a foundational toolkit for geometric
and structural analysis in materials science simulations.
"""
import copy
from typing import List, Tuple, Union, Any, Optional, Dict

import numpy as np
import fractions
import re
from scipy.spatial.distance import cdist
from tqdm import tqdm

from basic_function import unit_cell_parser
from basic_function import chemical_knowledge
from basic_function import operation


class Atom:
    """
    Represents a single atom in a chemical structure.

    Attributes:
        element (str): The chemical symbol of the atom (e.g., 'H', 'C', 'O').
        cart_xyz (np.ndarray): Cartesian coordinates [x, y, z] in Angstroms.
        frac_xyz (np.ndarray): Fractional coordinates [u, v, w] with respect to a lattice.
        atom_id (int): A unique identifier for the atom within a larger structure.
        force (np.ndarray): Force vector [fx, fy, fz] acting on the atom.
        atom_charge (float): Partial charge of the atom.
        atom_energy (float): Site potential energy of the atom.
        molecule (int): Identifier for the molecule this atom belongs to.
        bonded_atom (list): A list of IDs of atoms bonded to this one.
        descriptor (any): A placeholder for feature vectors or other descriptors.
        comment (dict): A dictionary for storing arbitrary metadata.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes an Atom object.

        Args:
            **kwargs: Keyword arguments to set atom attributes.
                Required: 'element' and one of 'cart_xyz' or 'frac_xyz'.
                Optional: 'atom_id', 'force_xyz', 'atom_charge', 'atom_energy', etc.
        """
        self.element: str = kwargs.get("element", "unknown")
        self.cart_xyz: Union[str, np.ndarray] = kwargs.get('cart_xyz', "unknown")
        self.frac_xyz: Union[str, np.ndarray] = kwargs.get('frac_xyz', "unknown")
        self.atom_id: Union[str, int] = kwargs.get("atom_id", "unknown")
        self.force: Union[str, np.ndarray] = kwargs.get('force_xyz', 'unknown')
        self.atom_charge: Union[str, float] = kwargs.get('atom_charge', 'unknown')
        self.atom_energy: Union[str, float] = kwargs.get('atom_energy', 'unknown')
        self.molecule: Union[str, int] = kwargs.get('molecule', 'unknown')
        self.bonded_atom: list = kwargs.get('bonded_atom', [])
        self.descriptor: Any = kwargs.get("descriptor", "unknown")
        self.comment: dict = kwargs.get("comment", {})

    def info(self) -> None:
        """Prints all attributes of the atom to the console."""
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

    def check(self) -> None:
        """
        Performs basic sanity checks on the atom's attributes.

        Raises:
            AssertionError: If element is not defined, or if neither cartesian
                            nor fractional coordinates are provided.
        """
        assert self.element != "unknown", "Atom must have an element type."
        has_cart = isinstance(self.cart_xyz, (np.ndarray, list))
        has_frac = isinstance(self.frac_xyz, (np.ndarray, list))
        assert has_cart or has_frac, "Atom needs either cart_xyz or frac_xyz."


class Crystal:
    """
    Represents a periodic crystal structure.

    Contains lattice information (cell parameters or vectors) and a list of atoms
    that constitute the structure within the unit cell.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes a Crystal object.

        The constructor requires lattice and atom information. It will automatically
        calculate derived properties like volume and density.

        Args:
            **kwargs: Keyword arguments to set crystal attributes.
                Required: 'atoms' and one of 'cell_vect' or 'cell_para'.
                - cell_vect (list): 3x3 list or array of lattice vectors.
                - cell_para (list): [[a, b, c], [alpha, beta, gamma]].
                - atoms (List[Atom]): A list of Atom objects.
                Optional: 'energy', 'comment', 'system_name', 'space_group', 'SYMM', etc.
        """
        self.cell_vect: Union[str, np.ndarray] = kwargs.get("cell_vect", "unknown")
        self.cell_para: Union[str, list] = kwargs.get("cell_para", "unknown")
        self.atoms: Union[str, List[Atom]] = kwargs.get("atoms", "unknown")
        self.energy: Union[str, float] = kwargs.get("energy", "unknown")
        self.comment: Any = kwargs.get("comment", "unknown")
        self.descriptor: Any = kwargs.get("descriptor", "unknown")
        self.molecule_number: Union[str, int] = kwargs.get("molecule_number", "unknown")
        self.system_name: str = kwargs.get("system_name", "unknown")
        self.virial: Any = kwargs.get("virial", "unknown")
        self.SYMM: list = kwargs.get("SYMM", ["x,y,z"])
        self.space_group: int = kwargs.get("space_group", 1)
        self.other_properties: dict = {}

        # This method completes the initialization.
        self.lattice_and_atom_complete()

    def lattice_and_atom_complete(self) -> None:
        """
        Completes the initialization by ensuring consistency between cell representations,
        atom coordinates, and calculating derived properties.
        """
        # --- 1. Finalize Lattice Representation ---
        has_vect = isinstance(self.cell_vect, (np.ndarray, list))
        has_para = isinstance(self.cell_para, (np.ndarray, list))

        if has_vect and has_para:
            # If both are provided, check for consistency
            derived_para = np.array(unit_cell_parser.cell_vect_to_para(self.cell_vect)).flatten()
            provided_para = np.array(self.cell_para).flatten()
            assert np.allclose(derived_para, provided_para, atol=1e-3), \
                "Provided cell_para and cell_vect are inconsistent."
        elif has_vect and not has_para:
            self.cell_para = unit_cell_parser.cell_vect_to_para(self.cell_vect)
        elif not has_vect and has_para:
            self.cell_vect = unit_cell_parser.cell_para_to_vect(self.cell_para)
        else:
            raise ValueError("Crystal lattice is not defined. Provide 'cell_vect' or 'cell_para'.")

        # --- 2. Finalize Atom Coordinates ---
        if self.atoms == "unknown" or not self.atoms:
            print("Warning: Crystal initialized with no atoms.")
            self.atoms = []
        else:
            for atom in self.atoms:
                has_cart = isinstance(atom.cart_xyz, (np.ndarray, list))
                has_frac = isinstance(atom.frac_xyz, (np.ndarray, list))
                if not (has_cart and has_frac):
                    if has_frac:
                        atom.cart_xyz = unit_cell_parser.atom_frac_to_cart_by_cell_vect(atom.frac_xyz, self.cell_vect)
                    elif has_cart:
                        atom.frac_xyz = unit_cell_parser.atom_cart_to_frac_by_cell_vect(atom.cart_xyz, self.cell_vect)
                    else:
                        raise ValueError(f"Atom {atom.element} {atom.atom_id} has no coordinate information.")

        # --- 3. Calculate Derived Properties ---
        self.volume = unit_cell_parser.calculate_volume(self.cell_para)
        if self.atoms:
            total_mass = sum(chemical_knowledge.element_masses[atom.element] for atom in self.atoms)
            # Density in g/cm^3
            self.density = total_mass / (self.volume * 1e-24) / (6.022140857e23)
        else:
            self.density = 0.0

    def update_cart_by_frac(self) -> None:
        """Updates all atom cartesian coordinates from their fractional coordinates."""
        for atom in self.atoms:
            atom.cart_xyz = unit_cell_parser.atom_frac_to_cart_by_cell_vect(atom.frac_xyz, self.cell_vect)

    def update_frac_by_cart(self) -> None:
        """Updates all atom fractional coordinates from their cartesian coordinates."""
        for atom in self.atoms:
            atom.frac_xyz = unit_cell_parser.atom_cart_to_frac_by_cell_vect(atom.cart_xyz, self.cell_vect)

    def check(self) -> None:
        """Performs consistency checks on the crystal structure."""
        print("Performing consistency checks...")
        # Check lattice consistency
        self.lattice_and_atom_complete()

        # Check atom coordinate consistency
        for atom in self.atoms:
            derived_cart = unit_cell_parser.atom_frac_to_cart_by_cell_vect(atom.frac_xyz, self.cell_vect)
            assert np.allclose(atom.cart_xyz, derived_cart, atol=1e-3), \
                f"Atom {atom.atom_id} cartesian and fractional coordinates do not match."

        # Check atom IDs
        if all(atom.atom_id != "unknown" for atom in self.atoms):
            print("All atoms have IDs.")
        else:
            print("Warning: Not all atoms have IDs. Use .give_atom_id_forced() to assign them.")
        print("Checks passed.")

    def give_atom_id_forced(self) -> None:
        """Assigns or resets atom IDs from 0 to N-1 and clears bonding info."""
        print("Warning: Resetting all atom IDs and bonding information!")
        for i, atom in enumerate(self.atoms):
            atom.atom_id = i
            atom.bonded_atom = []

    def move_atom_into_cell(self) -> None:
        """
        Moves all atoms into the primary unit cell [0, 1) in fractional coordinates.
        """
        for atom in self.atoms:
            # Use modulo for a more direct and efficient way to wrap coordinates
            atom.frac_xyz = np.mod(atom.frac_xyz, 1.0)
        self.update_cart_by_frac()

    def find_molecule(self, tolerance: float = 1.15) -> None:
        """
        Identifies molecules within the crystal based on bonding distances.

        This method performs a graph search (BFS) on the atoms, connecting them
        based on scaled covalent radii. It populates the `atom.molecule` and
        `self.molecule_number` attributes.

        Args:
            tolerance: A scaling factor for covalent radii to determine bonding.
                       A bond is formed if dist(A, B) < (radius(A) + radius(B)) * tolerance.
        """
        self.move_atom_into_cell()
        atoms_to_visit = list(range(len(self.atoms)))
        molecule_id = 0

        while atoms_to_visit:
            molecule_id += 1
            # Start a Breadth-First Search (BFS) from the first unvisited atom
            q = [atoms_to_visit[0]]
            visited_in_molecule = {atoms_to_visit[0]}

            head = 0
            while head < len(q):
                current_atom_idx = q[head]
                head += 1
                self.atoms[current_atom_idx].molecule = molecule_id
                
                # Check for bonds with all other atoms
                for other_atom_idx in range(len(self.atoms)):
                    if current_atom_idx == other_atom_idx:
                        continue
                    
                    # is_bonding_crystal handles periodic boundaries
                    is_bonded, _ = operation.is_bonding_crystal(
                        self.atoms[current_atom_idx],
                        self.atoms[other_atom_idx],
                        self.cell_vect,
                        tolerance=tolerance,
                        update_atom2=False # Do not modify coordinates during search
                    )

                    if is_bonded and other_atom_idx not in visited_in_molecule:
                        visited_in_molecule.add(other_atom_idx)
                        q.append(other_atom_idx)
            
            # Remove all atoms found in the new molecule from the list to visit
            atoms_to_visit = [idx for idx in atoms_to_visit if idx not in visited_in_molecule]

        self.molecule_number = molecule_id

    def get_element(self) -> List[str]:
        """Returns a sorted list of unique element symbols in the crystal."""
        return chemical_knowledge.sort_by_atomic_number(set(atom.element for atom in self.atoms))

    def get_element_amount(self) -> List[int]:
        """Returns the count of each element, sorted by atomic number."""
        all_elements = [atom.element for atom in self.atoms]
        return [all_elements.count(element) for element in self.get_element()]


    def make_p1(self) -> None:
        """
        Expands the asymmetric unit to the full P1 cell using symmetry operations.

        The crystal's space group is set to 1 (P1) and SYMM is reset. This
        implementation is robustly designed to ensure the final coordinate array
        is always 2-dimensional, preventing downstream errors.
        """
        all_ele, all_frac = self.get_ele_and_frac()
        all_reflect_position = []
        all_matrix_M = []
        all_matrix_C = []
        for sym_opt in self.SYMM:
            sym_opt_ele = sym_opt.lower().replace(" ", "").split(",")
            # assert len(sym_opt_ele) == 3, "sym {} could not be treat".format(sym_opt_ele)
            matrix_M = np.zeros((3, 3))
            matrix_C = np.zeros((1, 3))
            for idx, word in enumerate(sym_opt_ele):
                sym_opt_ele_split = re.findall(r".*?([+-]*[xyz0-9\/\.]+)", word)
                for sym_opt_frag in sym_opt_ele_split:
                    if sym_opt_frag == 'x' or sym_opt_frag == '+x':
                        matrix_M[0][idx] = 1
                    elif str(sym_opt_frag) == '-x':
                        matrix_M[0][idx] = -1
                    elif sym_opt_frag == 'y' or sym_opt_frag == '+y':
                        matrix_M[1][idx] = 1
                    elif sym_opt_frag == '-y':
                        matrix_M[1][idx] = -1
                    elif sym_opt_frag == 'z' or sym_opt_frag == '+z':
                        matrix_M[2][idx] = 1
                    elif sym_opt_frag == '-z':
                        matrix_M[2][idx] = -1
                    elif operation.is_number(sym_opt_frag) is True:
                        matrix_C[0][idx] = float(fractions.Fraction(sym_opt_frag))
                    else:
                        raise Exception("wrong sym opt of" + sym_opt_frag)

            all_matrix_M.append(matrix_M)
            all_matrix_C.append(matrix_C)

        for j in range(0, len(all_matrix_M)):
            new_positions = np.dot(np.array([all_frac]), all_matrix_M[j]) + all_matrix_C[j]
            all_reflect_position.append(new_positions.squeeze())
        all_ele = all_ele*len(self.SYMM)

        new_atoms = []
        idx=0
        for element, frac_xyz in zip(all_ele, np.array(all_reflect_position).reshape(-1,3)):
            new_atoms.append(Atom(element=element,
                                  frac_xyz=frac_xyz,
                                  atom_id=idx))
            idx+=1

        self.SYMM = "[x,y,z]"
        self.space_group = 1
        self.atoms = new_atoms
        self.update_cart_by_frac()

    def sort_by_element(self) -> None:
        """Sorts the atoms list based on atomic number."""
        self.atoms.sort(key=lambda atom: chemical_knowledge.periodic_table_list[atom.element])

    def get_ele_and_cart(self) -> Tuple[List[str], np.ndarray]:
        """Returns all element symbols and their cartesian coordinates."""
        if not self.atoms:
            return [], np.empty((0, 3))
        all_ele = [atom.element for atom in self.atoms]
        all_carts = np.array([atom.cart_xyz for atom in self.atoms])
        return all_ele, all_carts

    def get_ele_and_frac(self) -> Tuple[List[str], np.ndarray]:
        """Returns all element symbols and their fractional coordinates."""
        if not self.atoms:
            return [], np.empty((0, 3))
        all_ele = [atom.element for atom in self.atoms]
        all_fracs = np.array([atom.frac_xyz for atom in self.atoms])
        return all_ele, all_fracs

    def info(self, all_info: bool = False) -> None:
        """
        Prints a formatted summary of the crystal structure.

        Args:
            all_info: If True, prints an extended table including fractional
                      coordinates, forces, and other properties.
        """
        print("--- Crystal System ---")
        print(f"Name: {self.system_name}")
        print("Lattice Vectors (Angstrom):")
        for vec in self.cell_vect:
            print(f"{vec[0]:16.8f} {vec[1]:16.8f} {vec[2]:16.8f}")
        print("Lattice Parameters:")
        print(f"a, b, c (A): {self.cell_para[0][0]:.4f}, {self.cell_para[0][1]:.4f}, {self.cell_para[0][2]:.4f}")
        print(f"alpha, beta, gamma (deg): {self.cell_para[1][0]:.4f}, {self.cell_para[1][1]:.4f}, {self.cell_para[1][2]:.4f}")
        print(f"Volume (A^3): {self.volume:.4f} | Density (g/cm^3): {self.density:.4f}")
        print(f"\n--- Atomic Coordinates (Total: {len(self.atoms)}) ---")
        
        if not all_info:
            print(f"{'Element':<10} {'Cartesian X':>16} {'Cartesian Y':>16} {'Cartesian Z':>16}")
            print("-" * 58)
            for atom in self.atoms:
                print(f"{atom.element:<10} {atom.cart_xyz[0]:16.8f} {atom.cart_xyz[1]:16.8f} {atom.cart_xyz[2]:16.8f}")
        else:
            header = (
                f"{'ID':<5} {'Elem':<6} "
                f"{'Frac X':>10} {'Frac Y':>10} {'Frac Z':>10} | "
                f"{'Cart X':>12} {'Cart Y':>12} {'Cart Z':>12}"
            )
            print(header)
            print("-" * len(header))
            for atom in self.atoms:
                aid = str(atom.atom_id) if atom.atom_id != 'unknown' else '-'
                print(
                    f"{aid:<5} {atom.element:<6} "
                    f"{atom.frac_xyz[0]:10.6f} {atom.frac_xyz[1]:10.6f} {atom.frac_xyz[2]:10.6f} | "
                    f"{atom.cart_xyz[0]:12.6f} {atom.cart_xyz[1]:12.6f} {atom.cart_xyz[2]:12.6f}"
                )
            
            print("\n--- Other Properties ---")
            print(f"Energy: {self.energy}")
            print(f"Comment: {self.comment}")
            print(f"Virial: {self.virial}")


class Molecule:
    """Represents a non-periodic molecule (a collection of atoms)."""

    def __init__(self, **kwargs: Any):
        """
        Initializes a Molecule object.

        Args:
            **kwargs: Keyword arguments to set molecule attributes.
                Required: 'atoms' (List[Atom]).
                Optional: 'energy', 'comment', 'name', 'system_name'.
        """
        self.atoms: Union[str, List[Atom]] = kwargs.get("atoms", "unknown")
        self.energy: Union[str, float] = kwargs.get("energy", "unknown")
        self.comment: Any = kwargs.get("comment", "unknown")
        self.descriptor: Any = kwargs.get("descriptor", "unknown")
        self.name: str = kwargs.get("name", "unknown")
        self.system_name: str = kwargs.get("system_name", "unknown")

        if self.atoms == "unknown":
            print("Warning: Molecule initialized with no atoms.")
            self.atoms = []

    def give_atom_id_forced(self) -> None:
        """Assigns or resets atom IDs from 0 to N-1 and clears bonding info."""
        print("Warning: Resetting all atom IDs and bonding information!")
        for i, atom in enumerate(self.atoms):
            atom.atom_id = i
            atom.bonded_atom = []

    def get_element(self) -> List[str]:
        """Returns a sorted list of unique element symbols in the molecule."""
        if not self.atoms: return []
        return chemical_knowledge.sort_by_atomic_number(set(atom.element for atom in self.atoms))

    def get_element_amount(self) -> List[int]:
        """Returns the count of each element, sorted by atomic number."""
        if not self.atoms: return []
        all_elements = [atom.element for atom in self.atoms]
        return [all_elements.count(element) for element in self.get_element()]

    def get_ele_and_cart(self) -> Tuple[List[str], np.ndarray]:
        """Returns all element symbols and their cartesian coordinates."""
        if not self.atoms:
            return [], np.empty((0, 3))
        all_ele = [atom.element for atom in self.atoms]
        all_carts = np.array([atom.cart_xyz for atom in self.atoms])
        return all_ele, all_carts

    def put_ele_cart_back(self, all_ele: List[str], all_carts: np.ndarray) -> None:
        """Updates the molecule's atoms from lists of elements and coordinates."""
        for i, atom in enumerate(self.atoms):
            atom.element = all_ele[i]
            atom.cart_xyz = all_carts[i]

    def build_molecules_by_ele_cart(self, all_ele: List[str], all_carts: np.ndarray) -> None:
        """Rebuilds the molecule's atoms list from elements and coordinates."""
        assert len(all_ele) == len(all_carts), "Element and coordinate lists must have the same length."
        self.atoms = [
            Atom(element=ele, cart_xyz=cart, atom_id=i)
            for i, (ele, cart) in enumerate(zip(all_ele, all_carts))
        ]

    def get_mass(self) -> float:
        """Calculates the total mass of the molecule."""
        if not self.atoms: return 0.0
        return sum(chemical_knowledge.element_masses[atom.element] for atom in self.atoms)

    def get_center_of_mass(self) -> np.ndarray:
        """Calculates the center of mass of the molecule."""
        if not self.atoms: return np.zeros(3)
        
        all_ele, all_carts = self.get_ele_and_cart()
        masses = np.array([chemical_knowledge.element_masses[x] for x in all_ele])
        total_mass = np.sum(masses)
        
        if total_mass == 0: return np.zeros(3)
        return np.sum(all_carts * masses[:, np.newaxis], axis=0) / total_mass

    def sort_by_element(self) -> None:
        """Sorts the atoms list based on atomic number."""
        self.atoms.sort(key=lambda atom: chemical_knowledge.periodic_table_list[atom.element])

    def sort_by_id(self) -> None:
        """Sorts the atoms list based on their atom_id."""
        self.atoms.sort(key=lambda atom: atom.atom_id)

    def info(self) -> None:
        """Prints a formatted summary of the molecule."""
        print(f"--- Molecule ---")
        print(f"Name: {self.name} | System: {self.system_name}")
        print(f"Number of atoms: {len(self.atoms)}")
        print(f"Total Mass (amu): {self.get_mass():.4f}")
        print(f"Energy: {self.energy}")
        print(f"Comment: {self.comment}")
        print(f"\n{'Element':<10} {'Cartesian X':>16} {'Cartesian Y':>16} {'Cartesian Z':>16}")
        print("-" * 58)
        if self.atoms:
            for atom in self.atoms:
                print(f"{atom.element:<10} {atom.cart_xyz[0]:16.8f} {atom.cart_xyz[1]:16.8f} {atom.cart_xyz[2]:16.8f}")

    def find_fragment(self, tolerance: float = 1.15) -> Dict[int, List[int]]:
        """
        Identifies covalently bonded fragments within the molecule.

        This is useful for molecules that are actually composed of several
        disconnected components (e.g., salts, solvent shells).

        Args:
            tolerance: Scaling factor for covalent radii to determine bonding.

        Returns:
            A dictionary mapping a fragment ID (starting from 1) to a list of
            atom indices belonging to that fragment.
        """
        if not self.atoms: return {}

        num_atoms = len(self.atoms)
        cart_matrix = np.array([atom.cart_xyz for atom in self.atoms])
        radii = np.array([chemical_knowledge.element_covalent_radii[atom.element] for atom in self.atoms])
        
        # Create a matrix of bond thresholds (r_i + r_j)
        bond_threshold_matrix = (radii[:, np.newaxis] + radii) * tolerance
        
        # True where distance is less than the bond threshold
        dist_matrix = cdist(cart_matrix, cart_matrix)
        adj_matrix = dist_matrix < bond_threshold_matrix
        np.fill_diagonal(adj_matrix, False)

        # Graph traversal (DFS) to find connected components
        visited = [False] * num_atoms
        groups = {}
        group_id = 0
        for i in range(num_atoms):
            if not visited[i]:
                group_id += 1
                groups[group_id] = []
                stack = [i]
                while stack:
                    atom_idx = stack.pop()
                    if not visited[atom_idx]:
                        visited[atom_idx] = True
                        groups[group_id].append(atom_idx)
                        # Find neighbors and add to stack
                        neighbors = np.where(adj_matrix[atom_idx])[0]
                        stack.extend(neighbors)
        return groups

    def give_molecule_id(self, tolerance: float = 1.15) -> None:
        """Assigns a molecule ID to each atom based on fragment analysis."""
        fragments = self.find_fragment(tolerance=tolerance)
        for group_id, atom_indices in fragments.items():
            for atom_idx in atom_indices:
                self.atoms[atom_idx].molecule = group_id

    def take_out_fragment(self, tolerance: float = 1.15) -> List['Molecule']:
        """
        Splits the current molecule into a list of new Molecule objects,
        one for each disconnected fragment.
        """
        if not self.atoms: return []

        self.give_atom_id_forced() # Ensure IDs are set for lookup
        fragments = self.find_fragment(tolerance=tolerance)
        new_molecules = []
        
        for i, atom_indices in fragments.items():
            fragment_atoms = [self.atoms[j] for j in atom_indices]
            new_mol = Molecule(
                atoms=copy.deepcopy(fragment_atoms),
                name=f"{self.name}_frag{i}",
                system_name=f"{self.system_name}_frag{i}"
            )
            new_molecules.append(new_mol)
        return new_molecules

    def calculate_frac_xyz_by_cell_para(self, cell_para: list) -> None:
        """Calculates fractional coordinates for all atoms given cell parameters."""
        for atom in self.atoms:
            atom.frac_xyz = unit_cell_parser.atom_cart_to_frac_by_cell_para(atom.cart_xyz, cell_para)

    def molecule_volume(self, num_samples: int = 100000) -> float:
        """
        Calculates the van der Waals volume using a Monte Carlo integration method.

        This method samples points in a bounding box around the molecule and
        determines the ratio of points that fall within any atom's vdW sphere.

        Args:
            num_samples: The number of random points to sample. More points
                         yield a more accurate volume at the cost of performance.

        Returns:
            The estimated van der Waals volume in cubic Angstroms.
        """
        if not self.atoms: return 0.0

        elements, coords = self.get_ele_and_cart()
        radii = np.array([chemical_knowledge.element_vdw_radii[el] for el in elements])

        # Determine bounding box for sampling
        min_bounds = np.min(coords, axis=0) - np.max(radii)
        max_bounds = np.max(coords, axis=0) + np.max(radii)
        bounding_box_volume = np.prod(max_bounds - min_bounds)

        # Generate random sample points within the bounding box
        random_points = np.random.uniform(min_bounds, max_bounds, (num_samples, 3))

        # Check for each point if it's inside ANY sphere
        count_inside = 0
        for rp in tqdm(random_points, desc="Monte Carlo Volume", leave=False):
            # Calculate squared distances from the point to all atom centers
            dist_sq = np.sum((coords - rp)**2, axis=1)
            # If any distance is within the radius, the point is inside
            if np.any(dist_sq <= radii**2):
                count_inside += 1
        
        return (count_inside / num_samples) * bounding_box_volume