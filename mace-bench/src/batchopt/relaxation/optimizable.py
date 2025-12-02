"""
Copyright (c) Meta, Inc. and its affiliates.
Copyright (c) 2025 Ma Zhaojia

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Modified from original Meta implementation.
"""

from __future__ import annotations

from functools import cached_property
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, Any, Generator
import numpy as np
import torch
import logging
from ase.calculators.calculator import PropertyNotImplementedError
from ase.stress import voigt_6_to_full_3x3_stress
from torch_scatter import scatter

from batchopt.relaxation.ase_utils import batch_to_atoms


# Define dummy classes for when imports fail
class _DummyCalculator:
    pass


try:
    from mace.calculators import MACECalculator
except ImportError:
    logging.warning("Unable to import MACECalculator.")
    MACECalculator = _DummyCalculator

try:
    from chgnet.model.dynamics import CHGNetCalculator
except ImportError:
    logging.warning("Unable to import CHGNetCalculator.")
    CHGNetCalculator = _DummyCalculator

try:
    from sevenn.calculator import (
        SevenNetCalculator,
        SevenNetD3Calculator,
        D3Calculator,
    )
except ImportError:
    logging.warning("Unable to import SevenNetCalculator.")
    SevenNetCalculator = _DummyCalculator
    SevenNetD3Calculator = _DummyCalculator
    D3Calculator = _DummyCalculator

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
except ImportError:
    logging.warning("Unable to import FAIRChemCalculator.")
    FAIRChemCalculator = _DummyCalculator


# this can be removed after pinning ASE dependency >= 3.23
try:
    from ase.optimize.optimize import Optimizable
except ImportError:

    class Optimizable:
        pass


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from numpy.typing import NDArray
    from torch_geometric.data import Batch


ALL_CHANGES: set[str] = {
    "pos",
    "atomic_numbers",
    "cell",
    "pbc",
}


# @torch.compile
def compare_batches(
    batch1: Batch | None,
    batch2: Batch,
    tol: float = 1e-6,
    excluded_properties: set[str] | None = None,
) -> list[str]:
    """Compare properties between two batches

    Args:
        batch1: atoms batch
        batch2: atoms batch
        tol: tolerance used to compare equility of floating point properties
        excluded_properties: list of properties to exclude from comparison

    Returns:
        list of system changes, property names that are differente between batch1 and batch2
    """
    system_changes = []

    if batch1 is None:
        system_changes = ALL_CHANGES
    else:
        properties_to_check = set(ALL_CHANGES)
        if excluded_properties:
            properties_to_check -= set(excluded_properties)

        # Check properties that aren't
        for prop in ALL_CHANGES:
            if prop in properties_to_check:
                properties_to_check.remove(prop)
                if not torch.allclose(
                    getattr(batch1, prop), getattr(batch2, prop), atol=tol
                ):
                    system_changes.append(prop)

    return system_changes


class OptimizableBatch(Optimizable):
    """A Batch version of ase Optimizable Atoms

    This class can be used with ML relaxations in fairchem.core.relaxations.ml_relaxation
    or in ase relaxations classes, i.e. ase.optimize.lbfgs
    """

    ignored_changes: ClassVar[set[str]] = set()

    def __init__(
        self,
        batch: Batch,
        trainer: Any,  # Any calculator type (MACECalculator | CHGNetCalculator | SevenNetCalculator | FAIRChemCalculator)
        transform: torch.nn.Module | None = None,
        mask_converged: bool = True,
        numpy: bool = False,
        masked_eps: float = 1e-8,
        compute_stress: bool = False,
        use_fast_predict: bool = True,
        dtype: torch.dtype = torch.float64,
    ):
        """Initialize Optimizable Batch

        Args:
            batch: A batch of atoms graph data
            model: An instance of a BaseTrainer derived class
            transform: graph transform
            mask_converged: if true will mask systems in batch that are already converged
            numpy: whether to cast results to numpy arrays
            masked_eps: masking systems that are converged when using ASE optimizers results in divisions by zero
                from zero differences in masked positions at future steps, we add a small number to prevent this.
            compute_stress: whether to compute stress during prediction
            use_fast_predict: use fast prediction method when available
            dtype: data type for tensor operations (torch.float32 or torch.float64)
        """
        self.batch = batch.to(trainer.device)
        self.trainer = trainer
        self.transform = transform
        self.numpy = numpy
        self.mask_converged = mask_converged
        self._cached_batch = None
        self._update_mask = None
        self.torch_results = {}
        self.results = {}
        self._eps = masked_eps
        self.dtype = dtype

        self.otf_graph = True  # trainer._unwrapped_model.otf_graph
        if not self.otf_graph and "edge_index" not in self.batch:
            self.update_graph()

        self.batch.pos = self.batch.pos.to(dtype=self.dtype)
        self.batch.cell = self.batch.cell.to(dtype=self.dtype)
        self.compute_stress = compute_stress
        self.use_fast_predict = use_fast_predict

        # Determine calculator type once during initialization for efficiency
        self._calculator_type = self._determine_calculator_type()
        logging.info(
            f"OptimizableBatch initialized with calculator type: {self._calculator_type}"
        )

    def _determine_calculator_type(self) -> str:
        """Determine the type of calculator to avoid repeated isinstance checks."""
        # Check against actual imported classes, not dummy classes
        trainer_class_name = type(self.trainer).__name__
        trainer_module = type(self.trainer).__module__

        if (
            "mace" in trainer_module.lower()
            or trainer_class_name == "MACECalculator"
        ):
            return "mace"
        elif (
            "chgnet" in trainer_module.lower()
            or trainer_class_name == "CHGNetCalculator"
        ):
            return "chgnet"
        elif "sevenn" in trainer_module.lower() or trainer_class_name in [
            "SevenNetCalculator",
            "SevenNetD3Calculator",
            "D3Calculator",
        ]:
            return "sevennet"
        elif (
            "fairchem" in trainer_module.lower()
            or trainer_class_name == "FAIRChemCalculator"
        ):
            return "fairchem"
        else:
            return "default"

    @property
    def device(self):
        return self.trainer.device

    @property
    def batch_indices(self):
        """Get the batch indices specifying which position/force corresponds to which batch."""
        return self.batch.batch

    @property
    def converged_mask(self):
        if self._update_mask is not None:
            return torch.logical_not(self._update_mask)
        return None

    @property
    def update_mask(self):
        if self._update_mask is None:
            return torch.ones(len(self.batch), dtype=bool)
        return self._update_mask

    @property
    def converge_indices_list(self):
        return torch.where(~self.update_mask)[0].tolist()

    @property
    def elem_per_group(self):
        # This return value actually represents the number of elements
        # in a group within a batch. Each group corresponds to batch_indices.
        # It will count the number of CELL elements in each group.
        return torch.bincount(self.batch_indices)

    @property
    def batch_size(self):
        return len(torch.unique(self.batch_indices))

    def check_state(self, batch: Batch, tol: float = 1e-12) -> bool:
        """Check for any system changes since last calculation."""
        return compare_batches(
            self._cached_batch,
            batch,
            tol=tol,
            excluded_properties=set(self.ignored_changes),
        )

    def _predict(self) -> None:
        """Run prediction if batch has any changes."""
        # TODO: Currently, the batch inference interfaces of various models are not unified and are poorly implemented.
        system_changes = self.check_state(self.batch)
        if len(system_changes) > 0:
            if self._calculator_type == "mace":
                # FIXME: &&&
                # for key, val in self.batch.to_dict().items():
                #     print(f'&&& key: {key}, val: {val}')
                # self.torch_results = self.trainer.predict_debug(atoms_list, self.batch, compute_stress=self.compute_stress)
                # self.torch_results = self.trainer.predict(self.config_batch)
                if self.use_fast_predict:
                    self.torch_results = self.trainer.fast_predict(
                        self.batch, compute_stress=self.compute_stress
                    )
                    self.batch.pos = self.batch.pos.to(self.dtype)
                    self.batch.cell = self.batch.cell.to(self.dtype)
                else:
                    atoms_list = batch_to_atoms(
                        self.batch, results=None, wrap_pos=False, eps=1e-17
                    )
                    self.torch_results = self.trainer.predict(
                        atoms_list, compute_stress=self.compute_stress
                    )
            elif self._calculator_type == "fairchem":
                # TODO: FAIRChemCalculator does not support batch prediction yet
                atoms_list = batch_to_atoms(
                    self.batch, results=None, wrap_pos=False, eps=1e-17
                )
                self.torch_results = self.trainer.predict(atoms_list=atoms_list)
            elif self._calculator_type == "chgnet":
                atoms_list = batch_to_atoms(
                    self.batch, results=None, wrap_pos=False, eps=1e-17
                )
                model_prediction = self.trainer.predict(
                    atoms_list=atoms_list, task="efs"
                )
                results = {
                    "energy": torch.tensor(
                        [pred["e"].item() for pred in model_prediction],
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    "forces": torch.vstack(
                        [
                            torch.from_numpy(pred["f"]).to(
                                device=self.device, dtype=self.dtype
                            )
                            for pred in model_prediction
                        ]
                    ),
                    "stress": torch.vstack(
                        [
                            torch.from_numpy(pred["s"]).to(
                                device=self.device, dtype=self.dtype
                            )
                            for pred in model_prediction
                        ]
                    ).view(-1, 3, 3),
                }
                self.torch_results = results
            elif self._calculator_type == "sevennet":
                atoms_list = batch_to_atoms(
                    self.batch, results=None, wrap_pos=False, eps=1e-17
                )
                self.torch_results = self.trainer.predict(atoms_list=atoms_list)
            else:  # default case
                self.torch_results = self.trainer.predict(
                    self.batch, per_image=False, disable_tqdm=True
                )
            # save only subset of props in simple namespace instead of cloning the whole batch to save memory
            changes = ALL_CHANGES - set(self.ignored_changes)
            self._cached_batch = SimpleNamespace(
                **{prop: self.batch[prop].clone() for prop in changes}
            )

    def get_property(
        self, name, no_numpy: bool = False
    ) -> torch.Tensor | NDArray:
        """Get a predicted property by name."""
        self._predict()
        if self.numpy:
            self.results = {
                key: pred.item() if pred.numel() == 1 else pred.cpu().numpy()
                for key, pred in self.torch_results.items()
            }
        else:
            self.results = self.torch_results

        if name not in self.results:
            raise PropertyNotImplementedError(
                f"{name} not present in this calculation"
            )

        return (
            self.results[name]
            if no_numpy is False
            else self.torch_results[name]
        )

    def get_positions(self) -> torch.Tensor | NDArray:
        """Get the batch positions"""
        pos = self.batch.pos.clone()
        if self.numpy:
            if self.mask_converged:
                pos[~self.update_mask[self.batch.batch]] = self._eps
            pos = pos.cpu().numpy()

        return pos

    def set_positions(self, positions: torch.Tensor | NDArray) -> None:
        """Set the atom positions in the batch."""
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(
                positions, dtype=self.dtype, device=self.device
            )
        else:
            positions = positions.to(dtype=self.dtype, device=self.device)

        if self.mask_converged and self._update_mask is not None:
            mask = self.update_mask[self.batch.batch]
            self.batch.pos[mask] = positions[mask]
        else:
            self.batch.pos = positions

        if not self.otf_graph:
            self.update_graph()

    def get_forces(
        self, apply_constraint: bool = False, no_numpy: bool = False
    ) -> torch.Tensor | NDArray:
        """Get predicted batch forces."""
        forces = self.get_property("forces", no_numpy=no_numpy)
        if apply_constraint:
            fixed_idx = torch.where(self.batch.fixed == 1)[0]
            if isinstance(forces, np.ndarray):
                fixed_idx = fixed_idx.tolist()
            forces[fixed_idx] = 0.0
        return forces.view(-1, 3)

    def get_potential_energy(self, **kwargs) -> torch.Tensor | NDArray:
        """Get predicted energy as the sum of all batch energies."""
        # ASE 3.22.1 expects a check for force_consistent calculations
        if kwargs.get("force_consistent", False) is True:
            raise PropertyNotImplementedError(
                "force_consistent calculations are not implemented"
            )
        if (
            len(self.batch) == 1
        ):  # unfortunately batch size 1 returns a float, not a tensor
            return self.get_property("energy")
        return self.get_property("energy").sum()

    def get_potential_energies(self) -> torch.Tensor | NDArray:
        """Get the predicted energy for each system in batch."""
        return self.get_property("energy")

    def get_cells(self) -> torch.Tensor:
        """Get batch crystallographic cells."""
        return self.batch.cell

    def set_cells(
        self, cells: torch.Tensor | NDArray, scale_atoms=False
    ) -> None:
        """Set batch cells."""
        assert self.batch.cell.shape == cells.shape, "Cell shape mismatch"
        if isinstance(cells, np.ndarray):
            cells = torch.tensor(cells, dtype=self.dtype, device=self.device)
        cells = cells.to(dtype=self.dtype, device=self.device)
        if scale_atoms:
            from ase.geometry.cell import complete_cell

            # M = torch.linalg.solve(
            #     self.batch.cell.view(-1, 3, 3),
            #     cells.view(-1, 3, 3),
            # )
            # TODO: need to implement a sparse version.
            # tmp_pos = torch.matmul(self.batch.pos, M.reshape(-1,3))
            for i in range(self.batch_size):
                if not self.update_mask[i]:
                    continue
                M = np.linalg.solve(
                    complete_cell(self.batch.cell[i].cpu().detach().numpy()),
                    complete_cell(cells[i].cpu().detach().numpy()),
                )
                pos_update_mask = self.batch.batch == i
                self.batch.pos[pos_update_mask] = torch.matmul(
                    self.batch.pos[pos_update_mask],
                    torch.from_numpy(M).to(self.device).reshape(-1, 3),
                )
        self.batch.cell[self.update_mask] = cells[self.update_mask]

    def get_volumes(self) -> torch.Tensor:
        """Get a tensor of volumes for each cell in batch"""
        cells = self.get_cells()
        return torch.linalg.det(cells)

    def iterimages(self) -> Generator[Batch, None, None]:
        # XXX document purpose of iterimages - this is just needed to work with ASE optimizers
        yield self.batch

    def get_max_forces(
        self, forces: torch.Tensor | None = None, apply_constraint: bool = False
    ) -> torch.Tensor:
        """Get the maximum forces per structure in batch"""
        if forces is None:
            forces = self.get_forces(
                apply_constraint=apply_constraint, no_numpy=True
            )
        return scatter(
            (forces**2).sum(axis=1).sqrt(), self.batch_indices, reduce="max"
        )

    def converged(
        self,
        forces: torch.Tensor | NDArray | None,
        fmax: float,
        max_forces: torch.Tensor | None = None,
        f_upper_limit: float = 1e20,
    ) -> bool:
        """Check if norm of all predicted forces are below fmax"""
        if forces is not None:
            if isinstance(forces, np.ndarray):
                forces = torch.tensor(
                    forces, device=self.device, dtype=self.dtype
                )
            max_forces = self.get_max_forces(forces)
        elif max_forces is None:
            max_forces = self.get_max_forces()

        # Update mask is True for forces that are greater than fmax AND less than f_upper_limit
        update_mask = torch.logical_and(
            max_forces.ge(fmax), max_forces.le(f_upper_limit)
        )
        # update cached mask
        if self.mask_converged:
            if self._update_mask is None:
                self._update_mask = update_mask
            else:
                # some models can have random noise in their predictions, so the mask is updated by
                # keeping all previously converged structures masked even if new force predictions
                # push it slightly above threshold
                self._update_mask = torch.logical_and(
                    self._update_mask, update_mask
                )
            update_mask = self._update_mask

        return not torch.any(update_mask).item()

    def get_atoms_list(self) -> list[Atoms]:
        """Get ase Atoms objects corresponding to the batch"""
        self._predict()  # in case no predictions have been run
        return batch_to_atoms(self.batch, results=self.torch_results)

    def update_graph(self):
        """Update the graph if model does not use otf_graph."""
        graph = self.trainer._unwrapped_model.generate_graph(self.batch)
        self.batch.edge_index = graph.edge_index
        self.batch.cell_offsets = graph.cell_offsets
        self.batch.neighbors = graph.neighbors
        if self.transform is not None:
            self.batch = self.transform(self.batch)

    def __len__(self) -> int:
        # TODO: this might be changed in ASE to be 3 * len(self.atoms)
        return len(self.batch.pos)


class OptimizableUnitCellBatch(OptimizableBatch):
    """Modify the supercell and the atom positions in relaxations.

    Based on ase UnitCellFilter to work on data batches
    """

    def __init__(
        self,
        batch: Batch,
        trainer: Any,  # Any calculator type (MACECalculator | CHGNetCalculator | SevenNetD3Calculator | FAIRChemCalculator)
        transform: torch.nn.Module | None = None,
        numpy: bool = False,
        mask_converged: bool = True,
        mask: Sequence[bool] | None = None,
        cell_factor: float | torch.Tensor | None = None,
        hydrostatic_strain: bool = False,
        constant_volume: bool = False,
        scalar_pressure: float = 0.0,
        masked_eps: float = 1e-8,
        use_fast_predict: bool = True,
        dtype: torch.dtype = torch.float64,
    ):
        """Create a filter that returns the forces and unit cell stresses together, for simultaneous optimization.

        For full details see:
            E. B. Tadmor, G. S. Smith, N. Bernstein, and E. Kaxiras,
            Phys. Rev. B 59, 235 (1999)

        Args:
            batch: A batch of atoms graph data
            model: An instance of a BaseTrainer derived class
            transform: graph transform
            numpy: whether to cast results to numpy arrays
            mask_converged: if true will mask systems in batch that are already converged
            mask: a boolean mask specifying which strain components are allowed to relax
            cell_factor:
                Factor by which deformation gradient is multiplied to put
                it on the same scale as the positions when assembling
                the combined position/cell vector. The stress contribution to
                the forces is scaled down by the same factor. This can be thought
                of as a very simple preconditioner. Default is number of atoms
                which gives approximately the correct scaling.
            hydrostatic_strain:
                Constrain the cell by only allowing hydrostatic deformation.
                The virial tensor is replaced by np.diag([np.trace(virial)]*3).
            constant_volume:
                Project out the diagonal elements of the virial tensor to allow
                relaxations at constant volume, e.g. for mapping out an
                energy-volume curve. Note: this only approximately conserves
                the volume and breaks energy/force consistency so can only be
                used with optimizers that do require a line minimisation
                (e.g. FIRE).
            scalar_pressure:
                Applied pressure to use for enthalpy pV term. As above, this
                breaks energy/force consistency.
            masked_eps: masking systems that are converged when using ASE optimizers results in divisions by zero
                from zero differences in masked positions at future steps, we add a small number to prevent this.
            dtype: data type for tensor operations (torch.float32 or torch.float64)
        """
        super().__init__(
            batch=batch,
            trainer=trainer,
            transform=transform,
            numpy=numpy,
            mask_converged=mask_converged,
            masked_eps=masked_eps,
            compute_stress=True,
            use_fast_predict=use_fast_predict,
            dtype=dtype,
        )

        self.orig_cells = self.get_cells().clone()
        self.stress = None

        if mask is None:
            # mask = torch.eye(3, device=self.device)
            mask = torch.ones(6, device=self.device)

        # TODO make sure mask is on GPU
        if mask.shape == (6,):
            self.mask = torch.tensor(
                voigt_6_to_full_3x3_stress(mask.detach().cpu()),
                device=self.device,
            )
        elif mask.shape == (3, 3):
            self.mask = mask
        else:
            raise ValueError("shape of mask should be (3,3) or (6,)")

        if isinstance(cell_factor, float):
            cell_factor = cell_factor * torch.ones(
                (3 * len(batch), 1), requires_grad=False
            )
        if cell_factor is None:
            cell_factor = self.batch.natoms.repeat_interleave(3).unsqueeze(
                dim=1
            )

        self.hydrostatic_strain = hydrostatic_strain
        self.constant_volume = constant_volume
        self.pressure = scalar_pressure * torch.eye(3, device=self.device)
        self.cell_factor = cell_factor
        self.stress = None
        self._batch_trace = torch.vmap(torch.trace)
        self._batch_diag = torch.vmap(
            lambda x: x * torch.eye(3, device=x.device)
        )

    @cached_property
    def batch_indices(self):
        """Get the batch indices specifying which position/force corresponds to which batch.

        We augment this to specify the batch indices for augmented positions and forces.
        """
        augmented_batch = torch.repeat_interleave(
            torch.arange(
                len(self.batch),
                dtype=self.batch.batch.dtype,
                device=self.device,
            ),
            3,
        )
        return torch.cat([self.batch.batch, augmented_batch])

    def deform_grad(self):
        """Get the cell deformation matrix"""
        return torch.transpose(
            torch.linalg.solve(self.orig_cells, self.get_cells()), 1, 2
        )

    def get_positions(self):
        """Get positions and cell deformation gradient."""
        cur_deform_grad = self.deform_grad()
        natoms = self.batch.num_nodes
        pos = torch.zeros(
            (natoms + 3 * len(self.get_cells()), 3),
            dtype=self.batch.pos.dtype,
            device=self.device,
        )

        # Augmented positions are the self.atoms.positions but without the applied deformation gradient
        pos[:natoms] = torch.linalg.solve(
            cur_deform_grad[self.batch.batch, :, :],
            self.batch.pos.view(-1, 3, 1),
        ).view(-1, 3)
        # cell DOFs are the deformation gradient times a scaling factor
        pos[natoms:] = self.cell_factor * cur_deform_grad.view(-1, 3)
        return pos.cpu().numpy() if self.numpy else pos

    def set_positions(self, positions: torch.Tensor | NDArray) -> None:
        """Set positions and cell.

        positions has shape (natoms + ncells * 3, 3).
        the first natoms rows are the positions of the atoms, the last nsystems * three rows are the deformation tensor
        for each cell.
        """
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(
                positions, dtype=self.dtype, device=self.device
            )
        else:
            positions = positions.to(dtype=self.dtype, device=self.device)

        natoms = self.batch.num_nodes
        new_atom_positions = positions[:natoms]
        new_deform_grad = (positions[natoms:] / self.cell_factor).view(-1, 3, 3)

        # TODO check that in fact symmetry is preserved setting cells and positions
        # Set the new cell from the original cell and the new deformation gradient.  Both current and final structures
        # should preserve symmetry.
        new_cells = torch.bmm(
            self.orig_cells, torch.transpose(new_deform_grad, 1, 2)
        )
        self.set_cells(new_cells)

        # Set the positions from the ones passed in (which are without the deformation gradient applied) and the new
        # deformation gradient. This should also preserve symmetry
        new_atom_positions = torch.bmm(
            new_atom_positions.view(-1, 1, 3),
            torch.transpose(
                new_deform_grad[self.batch.batch, :, :].view(-1, 3, 3), 1, 2
            ),
        )
        super().set_positions(new_atom_positions.view(-1, 3))

    def get_potential_energy(self, **kwargs):
        """
        returns potential energy including enthalpy PV term.
        """
        atoms_energy = super().get_potential_energy(**kwargs)
        return atoms_energy + self.pressure[0, 0] * self.get_volumes().sum()

    def get_forces(
        self, apply_constraint: bool = False, no_numpy: bool = False
    ) -> torch.Tensor | NDArray:
        """Get forces and unit cell stress."""
        stress = self.get_property("stress", no_numpy=True).view(-1, 3, 3)
        atom_forces = self.get_property("forces", no_numpy=True)

        if apply_constraint:
            fixed_idx = torch.where(self.batch.fixed == 1)[0]
            atom_forces[fixed_idx] = 0.0

        volumes = self.get_volumes().view(-1, 1, 1)
        # virial = -volumes * stress + self.pressure.view(-1, 3, 3)
        virial = -volumes * (stress + self.pressure.view(-1, 3, 3))
        # print(f'&&& virial0: {virial}')
        cur_deform_grad = self.deform_grad()
        atom_forces = torch.bmm(
            atom_forces.view(-1, 1, 3),
            cur_deform_grad[self.batch.batch, :, :].view(-1, 3, 3),
        )
        virial = torch.linalg.solve(
            cur_deform_grad, torch.transpose(virial, dim0=1, dim1=2)
        )
        virial = torch.transpose(virial, dim0=1, dim1=2)

        # print(f'&&& virial1: {virial}')

        # TODO this does not work yet! maybe _batch_trace gives an issue
        if self.hydrostatic_strain:
            virial = self._batch_diag(self._batch_trace(virial) / 3.0)

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask.view(-1, 3, 3)

        if self.constant_volume:
            virial[:, range(3), range(3)] -= (
                self._batch_trace(virial).view(3, -1) / 3.0
            )

        natoms = self.batch.num_nodes
        augmented_forces = torch.zeros(
            (natoms + 3 * len(self.get_cells()), 3),
            device=self.device,
            dtype=atom_forces.dtype,
        )
        # print(f'&&& atom_forces: {atom_forces}')
        # print(f'&&& virial2: {virial}')
        augmented_forces[:natoms] = atom_forces.view(-1, 3)
        augmented_forces[natoms:] = virial.view(-1, 3) / self.cell_factor

        self.stress = -virial.view(-1, 9) / volumes.view(-1, 1)

        if self.numpy and not no_numpy:
            augmented_forces = augmented_forces.cpu().numpy()

        # print(f'&&& augmented_forces: {augmented_forces}')

        return augmented_forces

    def __len__(self):
        return len(self.batch.pos) + 3 * len(self.batch)

    def get_potential_energies(self) -> torch.Tensor:
        """Get the predicted energy for each system in batch."""
        return (
            self.get_property("energy").view(-1)
            + self.pressure[0, 0] * self.get_volumes()
        )
