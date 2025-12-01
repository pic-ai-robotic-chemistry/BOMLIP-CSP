"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import ast
import collections
import copy
import datetime
import errno
import functools
import importlib
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from bisect import bisect
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter, segment_coo, segment_csr

from torch_geometric.data.data import BaseData
from torch_geometric.data import Batch

# sort files by atomic number in descending order
def count_atoms_cif(file):
    in_atom_site = False 
    natoms = 0
    with open(file, 'r') as f:
        while line := f.readline():
            if line.lower().startswith("loop_"):
                in_atom_site = False 
                continue 
            # if line.lower().startswith("_atom_site_"):
            if "_atom_site_" in line.lower():
                in_atom_site = True 
                continue 
            if in_atom_site:
                if line.startswith("_"):
                    in_atom_site = False 
                    continue 
                elif line: 
                    natoms += 1
    return natoms 

# Override the collation method in `pytorch_geometric.data.InMemoryDataset`
def collate(data_list):
    keys = data_list[0].keys
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        data[key].append(item[key])
        if torch.is_tensor(item[key]):
            s = slices[key][-1] + item[key].size(item.__cat_dim__(key, item[key]))
        elif isinstance(item[key], (int, float)):
            s = slices[key][-1] + 1
        else:
            raise ValueError("Unsupported attribute type")
        slices[key].append(s)

    if hasattr(data_list[0], "__num_nodes__"):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)

    for key in keys:
        if torch.is_tensor(data_list[0][key]):
            data[key] = torch.cat(
                data[key], dim=data.__cat_dim__(key, data_list[0][key])
            )
        else:
            data[key] = torch.tensor(data[key])
        slices[key] = torch.tensor(slices[key], dtype=torch.long)

    return data, slices

def data_list_collater(
    data_list: list[BaseData], otf_graph: bool = False, to_dict: bool = False
) -> BaseData | dict[str, torch.Tensor]:
    batch = Batch.from_data_list(data_list)

    if not otf_graph:
        try:
            n_neighbors = []
            for _, data in enumerate(data_list):
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except (NotImplementedError, TypeError):
            logging.warning(
                "LMDB does not contain edge index information, set otf_graph=True"
            )

    if to_dict:
        batch = dict(batch.items())

    return batch