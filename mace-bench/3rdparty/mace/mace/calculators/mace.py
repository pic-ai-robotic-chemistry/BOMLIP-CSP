###########################################################################################
# The ASE Calculator for MACE
# Authors: Ilyes Batatia, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging

# pylint: disable=wrong-import-position
import os
from glob import glob
from pathlib import Path
from typing import List, Union

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from e3nn import o3

from mace import data
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils
from mace.tools.compile import prepare
from mace.tools.scripts_utils import extract_model
import random
from mace.tools.torch_geometric.batch import Batch

from mace.tools import (
    atomic_numbers_to_indices,
    to_one_hot,
)

import time


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Get the dtype of the model"""
    mode_dtype = next(model.parameters()).dtype
    if mode_dtype == torch.float64:
        return "float64"
    if mode_dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unknown dtype {mode_dtype}")


class MACECalculator(Calculator):
    """MACE ASE Calculator
    args:
        model_paths: str, path to model or models if a committee is produced
                to make a committee use a wild card notation like mace_*.model
        device: str, device to run on (cuda or cpu)
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        default_dtype: str, default dtype of model
        charges_key: str, Array field of atoms object where atomic charges are stored
        model_type: str, type of model to load
                    Options: [MACE, DipoleMACE, EnergyDipoleMACE]

    Dipoles are returned in units of Debye
    """

    def __init__(
        self,
        model_paths: Union[list, str, None] = None,
        models: Union[List[torch.nn.Module], torch.nn.Module, None] = None,
        device: str = "cpu",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="",
        charges_key="Qs",
        model_type="MACE",
        compile_mode=None,
        fullgraph=True,
        enable_cueq=False,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.device = device
        self.dtype=None
        if enable_cueq:
            assert model_type == "MACE", "CuEq only supports MACE models"
            compile_mode = None
        if "model_path" in kwargs:
            deprecation_message = (
                "'model_path' argument is deprecated, please use 'model_paths'"
            )
            if model_paths is None:
                logging.warning(f"{deprecation_message} in the future.")
                model_paths = kwargs["model_path"]
            else:
                raise ValueError(
                    f"both 'model_path' and 'model_paths' given, {deprecation_message} only."
                )

        if (model_paths is None) == (models is None):
            raise ValueError(
                "Exactly one of 'model_paths' or 'models' must be provided"
            )

        self.results = {}

        self.model_type = model_type
        self.compute_atomic_stresses = False

        if model_type == "MACE":
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
            ]
            if kwargs.get("compute_atomic_stresses", False):
                self.implemented_properties.extend(["stresses", "virials"])
                self.compute_atomic_stresses = True
        elif model_type == "DipoleMACE":
            self.implemented_properties = ["dipole"]
        elif model_type == "EnergyDipoleMACE":
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
                "dipole",
            ]
        else:
            raise ValueError(
                f"Give a valid model_type: [MACE, DipoleMACE, EnergyDipoleMACE], {model_type} not supported"
            )

        if model_paths is not None:
            if isinstance(model_paths, str):
                # Find all models that satisfy the wildcard (e.g. mace_model_*.pt)
                model_paths_glob = glob(model_paths)

                if len(model_paths_glob) == 0:
                    raise ValueError(f"Couldn't find MACE model files: {model_paths}")

                model_paths = model_paths_glob
            elif isinstance(model_paths, Path):
                model_paths = [model_paths]

            if len(model_paths) == 0:
                raise ValueError("No mace file names supplied")
            self.num_models = len(model_paths)

            # Load models from files
            self.models = [
                torch.load(f=model_path, map_location=device)
                for model_path in model_paths
            ]

        elif models is not None:
            if not isinstance(models, list):
                models = [models]

            if len(models) == 0:
                raise ValueError("No models supplied")

            self.models = models
            self.num_models = len(models)

        if self.num_models > 1:
            print(f"Running committee mace with {self.num_models} models")

            if model_type in ["MACE", "EnergyDipoleMACE"]:
                self.implemented_properties.extend(
                    ["energies", "energy_var", "forces_comm", "stress_var"]
                )
            elif model_type == "DipoleMACE":
                self.implemented_properties.extend(["dipole_var"])

        if compile_mode is not None:
            print(f"Torch compile is enabled with mode: {compile_mode}")
            self.models = [
                torch.compile(
                    prepare(extract_model)(model=model, map_location=device),
                    mode=compile_mode,
                    fullgraph=fullgraph,
                )
                for model in self.models
            ]
            self.use_compile = True
        else:
            self.use_compile = False

        # Ensure all models are on the same device
        for model in self.models:
            model.to(device)

        r_maxs = [model.r_max.cpu() for model in self.models]
        r_maxs = np.array(r_maxs)
        if not np.all(r_maxs == r_maxs[0]):
            raise ValueError(f"committee r_max are not all the same {' '.join(r_maxs)}")
        self.r_max = float(r_maxs[0])

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.models[0].atomic_numbers]
        )
        self.charges_key = charges_key

        try:
            self.available_heads: List[str] = self.models[0].heads  # type: ignore
        except AttributeError:
            self.available_heads = ["Default"]
        kwarg_head = kwargs.get("head", None)
        if kwarg_head is not None:
            self.head = kwarg_head
        else:
            self.head = [head for head in self.available_heads if head.lower() == "default"]
            if len(self.head) == 0:
                raise ValueError(
                    "Head keyword was not provided, and no head in the model is 'default'. "
                    "Please provide a head keyword to specify the head you want to use. "
                    f"Available heads are: {self.available_heads}"
                )
            self.head = self.head[0]

        print("Using head", self.head, "out of", self.available_heads)

        model_dtype = get_model_dtype(self.models[0])
        if default_dtype == "":
            print(
                f"No dtype selected, switching to {model_dtype} to match model dtype."
            )
            default_dtype = model_dtype
        if model_dtype != default_dtype:
            print(
                f"Default dtype {default_dtype} does not match model dtype {model_dtype}, converting models to {default_dtype}."
            )
            if default_dtype == "float64":
                self.models = [model.double() for model in self.models]
            elif default_dtype == "float32":
                self.models = [model.float() for model in self.models]
        torch_tools.set_default_dtype(default_dtype)
        if enable_cueq:
            print("Converting models to CuEq for acceleration")
            self.models = [
                run_e3nn_to_cueq(model, device=device).to(device)
                for model in self.models
            ]
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.dtype = torch.float64 if default_dtype == "float64" else torch.float32
        
        self.model_time = 0.0
        self.calc_time = 0.0

    def _create_result_tensors(
        self, model_type: str, num_models: int, num_atoms: int
    ) -> dict:
        """
        Create tensors to store the results of the committee
        :param model_type: str, type of model to load
            Options: [MACE, DipoleMACE, EnergyDipoleMACE]
        :param num_models: int, number of models in the committee
        :return: tuple of torch tensors
        """
        dict_of_tensors = {}
        if model_type in ["MACE", "EnergyDipoleMACE"]:
            energies = torch.zeros(num_models, device=self.device)
            node_energy = torch.zeros(num_models, num_atoms, device=self.device)
            forces = torch.zeros(num_models, num_atoms, 3, device=self.device)
            stress = torch.zeros(num_models, 3, 3, device=self.device)
            dict_of_tensors.update(
                {
                    "energies": energies,
                    "node_energy": node_energy,
                    "forces": forces,
                    "stress": stress,
                }
            )
        if model_type in ["EnergyDipoleMACE", "DipoleMACE"]:
            dipole = torch.zeros(num_models, 3, device=self.device)
            dict_of_tensors.update({"dipole": dipole})
        return dict_of_tensors

    def _atoms_to_batch(self, atoms):
        keyspec = data.KeySpecification(
            info_keys={}, arrays_keys={"charges": self.charges_key}
        )
        config = data.config_from_atoms(
            atoms, key_specification=keyspec, head_name=self.head
        )
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config,
                    z_table=self.z_table,
                    cutoff=self.r_max,
                    heads=self.available_heads,
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)
        return batch

    def _clone_batch(self, batch):
        batch_clone = batch.clone()
        if self.use_compile:
            batch_clone["node_attrs"].requires_grad_(True)
            batch_clone["positions"].requires_grad_(True)
        return batch_clone

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        calc_start_t = time.perf_counter()
        Calculator.calculate(self, atoms)

        batch_base = self._atoms_to_batch(atoms)

        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            batch = self._clone_batch(batch_base)
            node_heads = batch["head"][batch["batch"]]
            num_atoms_arange = torch.arange(batch["positions"].shape[0])
            node_e0 = self.models[0].atomic_energies_fn(batch["node_attrs"])[
                num_atoms_arange, node_heads
            ]
            compute_stress = not self.use_compile
        else:
            compute_stress = False

        ret_tensors = self._create_result_tensors(
            self.model_type, self.num_models, len(atoms)
        )
        for i, model in enumerate(self.models):
            batch = self._clone_batch(batch_base)
            # print(f'@@@File: {__file__}, batch.to_dict(): {batch.to_dict()}')
            # set_seed(0)
            model_start_t = time.perf_counter()
            out = model(
                batch.to_dict(),
                compute_stress=compute_stress,
                training=self.use_compile,
                compute_edge_forces=self.compute_atomic_stresses,
                compute_atomic_stresses=self.compute_atomic_stresses,
            )
            model_end_t = time.perf_counter()
            self.model_time += (model_end_t - model_start_t)
            # print(f'&&& batch.positions: {batch["positions"]}')
            # print(f'&&& batch.stress: {batch["stress"]}')
            # print(f'compute_stress: {compute_stress}')
            # for k,v in batch.to_dict().items():
            #     print(f'&&& batch.to_dict(): {k} {v}')
            # print("=======")
            # print(f'&&& out["forces"]: {out["forces"]}')
            # print(f'&&& training: {self.use_compile}')
            # print(f'@@@File: {__file__}, out: {out}')
            if self.model_type in ["MACE", "EnergyDipoleMACE"]:
                ret_tensors["energies"][i] = out["energy"].detach()
                ret_tensors["node_energy"][i] = (out["node_energy"] - node_e0).detach()
                ret_tensors["forces"][i] = out["forces"].detach()
                if out["stress"] is not None:
                    ret_tensors["stress"][i] = out["stress"].detach()
            if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
                ret_tensors["dipole"][i] = out["dipole"].detach()
            if self.model_type in ["MACE"]:
                if out["atomic_stresses"] is not None:
                    ret_tensors.setdefault("atomic_stresses", []).append(
                        out["atomic_stresses"].detach()
                    )
                if out["atomic_virials"] is not None:
                    ret_tensors.setdefault("atomic_virials", []).append(
                        out["atomic_virials"].detach()
                    )

        self.results = {}
        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            self.results["energy"] = (
                torch.mean(ret_tensors["energies"], dim=0).cpu().item()
                * self.energy_units_to_eV
            )
            self.results["free_energy"] = self.results["energy"]
            self.results["node_energy"] = (
                torch.mean(ret_tensors["node_energy"], dim=0).cpu().numpy()
            )
            self.results["forces"] = (
                torch.mean(ret_tensors["forces"], dim=0).cpu().numpy()
                * self.energy_units_to_eV
                / self.length_units_to_A
            )
            if self.num_models > 1:
                self.results["energies"] = (
                    ret_tensors["energies"].cpu().numpy() * self.energy_units_to_eV
                )
                self.results["energy_var"] = (
                    torch.var(ret_tensors["energies"], dim=0, unbiased=False)
                    .cpu()
                    .item()
                    * self.energy_units_to_eV
                )
                self.results["forces_comm"] = (
                    ret_tensors["forces"].cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A
                )
            if out["stress"] is not None:
                self.results["stress"] = full_3x3_to_voigt_6_stress(
                    torch.mean(ret_tensors["stress"], dim=0).cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A**3
                )
                if self.num_models > 1:
                    self.results["stress_var"] = full_3x3_to_voigt_6_stress(
                        torch.var(ret_tensors["stress"], dim=0, unbiased=False)
                        .cpu()
                        .numpy()
                        * self.energy_units_to_eV
                        / self.length_units_to_A**3
                    )
            if "atomic_stresses" in ret_tensors:
                self.results["stresses"] = (
                    torch.mean(torch.stack(ret_tensors["atomic_stresses"]), dim=0)
                    .cpu()
                    .numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A**3
                )
            if "atomic_virials" in ret_tensors:
                self.results["virials"] = (
                    torch.mean(torch.stack(ret_tensors["atomic_virials"]), dim=0)
                    .cpu()
                    .numpy()
                    * self.energy_units_to_eV
                )
        if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
            self.results["dipole"] = (
                torch.mean(ret_tensors["dipole"], dim=0).cpu().numpy()
            )
            if self.num_models > 1:
                self.results["dipole_var"] = (
                    torch.var(ret_tensors["dipole"], dim=0, unbiased=False)
                    .cpu()
                    .numpy()
                )
        
        calc_end_t = time.perf_counter()
        self.calc_time += (calc_end_t - calc_start_t)

    def get_hessian(self, atoms=None):
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        batch = self._atoms_to_batch(atoms)
        hessians = [
            model(
                self._clone_batch(batch).to_dict(),
                compute_hessian=True,
                compute_stress=False,
                training=self.use_compile,
            )["hessian"]
            for model in self.models
        ]
        hessians = [hessian.detach().cpu().numpy() for hessian in hessians]
        if self.num_models == 1:
            return hessians[0]
        return hessians

    def get_descriptors(self, atoms=None, invariants_only=True, num_layers=-1):
        """Extracts the descriptors from MACE model.
        :param atoms: ase.Atoms object
        :param invariants_only: bool, if True only the invariant descriptors are returned
        :param num_layers: int, number of layers to extract descriptors from, if -1 all layers are used
        :return: np.ndarray (num_atoms, num_interactions, invariant_features) of invariant descriptors if num_models is 1 or list[np.ndarray] otherwise
        """
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        num_interactions = int(self.models[0].num_interactions)
        if num_layers == -1:
            num_layers = num_interactions
        batch = self._atoms_to_batch(atoms)
        descriptors = [model(batch.to_dict())["node_feats"] for model in self.models]

        irreps_out = o3.Irreps(str(self.models[0].products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
        per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
        per_layer_features[-1] = (
            num_invariant_features  # Equivariant features not created for the last layer
        )

        if invariants_only:
            descriptors = [
                extract_invariant(
                    descriptor,
                    num_layers=num_layers,
                    num_features=num_invariant_features,
                    l_max=l_max,
                )
                for descriptor in descriptors
            ]
        to_keep = np.sum(per_layer_features[:num_layers])
        descriptors = [
            descriptor[:, :to_keep].detach().cpu().numpy() for descriptor in descriptors
        ]

        if self.num_models == 1:
            return descriptors[0]
        return descriptors


    def predict(self, atoms_list, compute_stress=False): 
        predictions = {'energy': [], 'forces': []}

        configs = [data.config_from_atoms(atoms, charges_key=self.charges_key) for atoms in atoms_list]
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max, heads=self.heads
                )
                for config in configs
            ],
            batch_size=len(atoms_list),
            shuffle=False,
            drop_last=False,
        )

        # get the first batch of data_loader
        batch_base = next(iter(data_loader)).to(self.device)

        # calculate node_e0
        # batch = self._clone_batch(batch_base)
        # node_heads = batch["head"][batch["batch"]]
        # num_atoms_arange = torch.arange(batch["positions"].shape[0])
        # node_e0 = self.models[0].atomic_energies_fn(batch["node_attrs"])[
        #     num_atoms_arange, node_heads
        # ]

        # set_seed(0)
        out = self.models[0](
            batch_base.to_dict(),
            compute_stress=compute_stress, # TODO: DO WE NEED TO COMPUTE STRESS?
            training=self.use_compile,
        )
        # print(f'&&& batch.positions: {batch["positions"]}')
        # print(f'&&& batch.stress: {batch["stress"]}')
        #     print(f'&&& batch.to_dict(): {k} {v}')
        # print("=======")
        # print(f'&&& out["forces"]: {out["forces"]}')
        # print(f'&&& training: {self.use_compile}')
        predictions["energy"] = out["energy"].unsqueeze(-1).detach()
        predictions["forces"] = out["forces"].detach()
        if compute_stress:
            predictions["stress"] = out["stress"].detach()

        # print(f'&&& predictions["forces"] in predict: {predictions["forces"]}')

        return predictions

    def fast_predict(self, gbatch, compute_stress=False):
        gbatch.pos = gbatch.pos.to(self.dtype)
        gbatch.cell = gbatch.cell.to(self.dtype)

        predictions = {'energy': [], 'forces': []}
        batch_base = self.convert_batch(gbatch)
        out = self.models[0](
            batch_base.to_dict(),
            compute_stress=compute_stress,
            training=self.use_compile,
        )
        predictions["energy"] = out["energy"].unsqueeze(-1).detach().to(torch.float64)
        predictions["forces"] = out["forces"].detach().to(torch.float64)
        if compute_stress:
            predictions["stress"] = out["stress"].detach().to(torch.float64)

        return predictions


    def convert_batch(self, gbatch): 
        from batchopt import radius_graph_pbc_cuda
        # edge_indices, cell_offsets, num_neighbors = radius_graph_pbc_mem_effi(
        # from batchopt.pbc_graph_legacy import radius_graph_pbc
        # edge_indices, cell_offsets, num_neighbors = radius_graph_pbc(
        edge_indices, cell_offsets, num_neighbors = radius_graph_pbc_cuda(
            gbatch,
            radius=4.5, 
            max_num_neighbors_threshold=float('inf'), 
            pbc=[True, True, True], 
            dtype=self.dtype
        )

        tmp = edge_indices[0].clone()
        edge_indices[0] = edge_indices[1]
        edge_indices[1] = tmp

        # Create a one-hot matrix with number of columns equal to max atomic number + 1
        indices = atomic_numbers_to_indices(gbatch["atomic_numbers"].to("cpu"), z_table=self.z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(self.z_table),
        ).to(self.device)

        cbatch = Batch(
            positions = gbatch["pos"].clone(),
            cell = gbatch["cell"].view(-1, 3),
            batch = gbatch["batch"],
            ptr = gbatch["ptr"],
            edge_index = edge_indices,
            unit_shifts = cell_offsets,
            node_attrs = one_hot,
        ) 

        return cbatch


    def predict_debug(self, atoms_list, gbatch, compute_stress=False): 
        predictions = {'energy': [], 'forces': []}

        configs = [data.config_from_atoms(atoms, charges_key=self.charges_key) for atoms in atoms_list]
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max, heads=self.heads
                )
                for config in configs
            ],
            batch_size=len(atoms_list),
            shuffle=False,
            drop_last=False,
        )

        # get the first batch of data_loader
        # batch_base = next(iter(data_loader)).to(self.device)
        batch_base_tmp = next(iter(data_loader)).to(self.device)
        batch2 = self.convert_batch(gbatch)
        batch_base = Batch(
            # positions = batch_base_tmp["positions"],
            positions = batch2["positions"],
            # node_attrs = batch_base_tmp["node_attrs"], 
            node_attrs = batch2["node_attrs"], 
            # cell = batch_base_tmp["cell"],
            cell = batch2["cell"],
            edge_index = batch2["edge_index"],
            unit_shifts = batch2["unit_shifts"],
            # batch = batch_base_tmp["batch"],
            batch = batch2["batch"],
            # ptr = batch_base_tmp["ptr"],
            ptr = batch2["ptr"],
        )

        torch.set_printoptions(threshold=float('inf'))

        # print(f'batch2["edge_index"]: {batch2["edge_index"]}')
        # print(f'batch2["unit_shifts"]: {batch2["unit_shifts"]}')
        # print(f'batch_base_tmp["edge_index"]: {batch_base_tmp["edge_index"]}')
        # print(f'batch_base_tmp["unit_shifts"]: {batch_base_tmp["unit_shifts"]}')

        # calculate node_e0
        # batch = self._clone_batch(batch_base)
        # node_heads = batch["head"][batch["batch"]]
        # num_atoms_arange = torch.arange(batch["positions"].shape[0])
        # node_e0 = self.models[0].atomic_energies_fn(batch["node_attrs"])[
        #     num_atoms_arange, node_heads
        # ]

        # set_seed(0)
        out = self.models[0](
            batch_base.to_dict(),
            compute_stress=compute_stress, # TODO: DO WE NEED TO COMPUTE STRESS?
            training=self.use_compile,
        )
        # print(f'&&& batch.positions: {batch["positions"]}')
        # print(f'&&& batch.cell: {batch["cell"]}')
        # print(f'&&& batch.stress: {batch["stress"]}')
        # for k,v in batch.to_dict().items():
        #     print(f'&&& batch.to_dict(): {k} {v}')
        # print("=======")
        # print(f'&&& out["forces"]: {out["forces"]}')
        # print(f'&&& training: {self.use_compile}')
        predictions["energy"] = out["energy"].unsqueeze(-1).detach()
        predictions["forces"] = out["forces"].detach()
        if compute_stress:
            predictions["stress"] = out["stress"].detach()

        # print(f'&&& predictions["forces"] in predict: {predictions["forces"]}')

        return predictions