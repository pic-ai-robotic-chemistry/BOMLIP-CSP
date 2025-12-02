
"""
Copyright (c) 2025 Ma Zhaojia

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import torch
from torch_scatter import scatter

from ..optimizable import OptimizableBatch

class BFGS:
    def __init__(
        self,
        optimizable_batch: OptimizableBatch,
        maxstep: float = 0.2,
        alpha: float = 70.0,
        early_stop = False,
    ) -> None:
        """
        Args:
        """
        self.optimizable = optimizable_batch
        self.maxstep = maxstep
        self.alpha = alpha
        # self.H0 = 1.0 / self.alpha
        self.trajectories = None
        self.device=self.optimizable.device

        self.fmax = None
        self.steps = None

        self.initialize()
        self.early_stop = early_stop
    
    
    def initialize(self):
        # initial hessian
        self.H0 = [
            torch.eye(3 * size, device=self.optimizable.device, dtype=torch.float64) * self.alpha 
            for size in self.optimizable.elem_per_group
        ]

        self.H = [None] * self.optimizable.batch_size
        self.pos0 = torch.zeros_like(self.optimizable.get_positions().reshape(-1), device=self.device, dtype=torch.float64)
        self.forces0 = torch.zeros_like(self.pos0, device=self.device, dtype=torch.float64)

    def restart_from_earlystop(self, restart_indices, old_batch_indices):
        H_new = []
        pos0_new = torch.zeros_like(self.optimizable.get_positions().reshape(-1), device=self.device, dtype=torch.float64)
        forces0_new = torch.zeros_like(pos0_new, device=self.device, dtype=torch.float64)

        # collect the preserved historical data by old_batch_indices
        for i, idx in enumerate(restart_indices):
            mask_old = (idx==old_batch_indices.repeat_interleave(3))
            mask = (i==self.optimizable.batch_indices.repeat_interleave(3))
            H_new.append(self.H[idx])
            pos0_new[mask] = self.pos0[mask_old]
            forces0_new[mask] = self.forces0[mask_old]

        # append new info for the new batch
        for i in range(len(H_new), self.optimizable.batch_size):
            H_new.append(None)

        self.H = H_new
        self.pos0 = pos0_new
        self.forces0 = forces0_new
        

    def run(self, fmax, maxstep, is_restart_earlystop=False, restart_indices=None, old_batch_indices=None):
        logging.info("Enter bfgs's main program.")
        self.fmax = fmax
        self.max_iter = maxstep

        if is_restart_earlystop:
            self.restart_from_earlystop(restart_indices, old_batch_indices)

        iteration = 0
        max_forces = self.optimizable.get_max_forces(apply_constraint=True)
        logging.info("Step   Fmax(eV/A)")

        while iteration < self.max_iter and not self.optimizable.converged(
            forces=None, fmax=self.fmax, max_forces=max_forces, f_upper_limit=1e25,
        ):
            if self.early_stop and iteration > 0:
                converge_indices = self.optimizable.converge_indices_list
                if len(converge_indices) > 0:
                    logging.info(f"Early stopping at iteration {iteration}")
                    break

            logging.info(
                f"{iteration} " + " ".join(f"{x:18.15g}" for x in max_forces.tolist())
            )

            self.step()
            max_forces = self.optimizable.get_max_forces(apply_constraint=True)
            iteration += 1

        logging.info(
            f"{iteration} " + " ".join(f"{x:18.15g}" for x in max_forces.tolist())
        )

        # GPU memory usage as per nvidia-smi seems to gradually build up as
        # batches are processed. This releases unoccupied cached memory.
        torch.cuda.empty_cache()

        # set predicted values to batch
        for name, value in self.optimizable.results.items():
            setattr(self.optimizable.batch, name, value)

        self.nsteps = iteration

        if self.early_stop:
            converge_indices_list = self.optimizable.converge_indices_list
            return converge_indices_list
        else:
            return self.optimizable.converged(
                forces=None, fmax=self.fmax, max_forces=max_forces
            )
        

    def step(self): 
        forces = self.optimizable.get_forces(apply_constraint=True).to(
            dtype=torch.float64
        )
        pos = self.optimizable.get_positions().to(dtype=torch.float64)
        dpos, steplengths = self.prepare_step(pos, forces)
        dpos = self.determine_step(dpos, steplengths)
        self.optimizable.set_positions(pos+dpos)


    def prepare_step(self, pos, forces):
        forces = forces.reshape(-1)
        pos = pos.view(-1)
        self.update(pos, forces, self.pos0, self.forces0)

        dpos_list = []
        cur_indices = self.optimizable.batch_indices.repeat_interleave(3)
        # 预初始化结果列表
        dpos_list = [None] * len(self.H)
        
        # 分离计算任务：仅对需要计算的H矩阵创建流
        calc_indices = [i for i, need_update in enumerate(self.optimizable.update_mask) if need_update]
        streams = [torch.cuda.Stream() for _ in calc_indices]
        
        # 并行执行实际计算
        for i, stream in zip(calc_indices, streams):
            with torch.cuda.stream(stream):
                omega, V = torch.linalg.eigh(self.H[i])
                dpos_list[i] = (V @ (forces[cur_indices==i].t() @ V / torch.abs(omega)).t())

        # 同步所有计算流
        torch.cuda.current_stream().synchronize()
        
        # 在主线程处理零张量
        for i in range(len(self.H)):
            if not self.optimizable.update_mask[i]:
                dpos_list[i] = torch.zeros_like(forces[cur_indices==i])

        # 同步所有流
        for stream in streams:
            stream.synchronize()
        
        # dpos = torch.vstack(dpos_list)
        dpos = torch.zeros_like(forces)
        for i in torch.unique(cur_indices):
            mask = (cur_indices == i)
            dpos[mask] = dpos_list[i]
        dpos = dpos.reshape(-1, 3)

        steplengths = (dpos ** 2).sum(dim=-1).sqrt()
        self.pos0 = pos
        self.forces0 = forces

        return dpos, steplengths


    def determine_step(self, dpos, steplengths):
        longest_steps = scatter(
            steplengths, self.optimizable.batch_indices, reduce="max"
        )
        longest_steps = longest_steps[self.optimizable.batch_indices]
        maxstep = longest_steps.new_tensor(self.maxstep)
        scale = (longest_steps).reciprocal() * torch.min(longest_steps, maxstep)
        dpos *= scale.unsqueeze(1)
        return dpos

    def update(self, pos, forces, pos0, forces0):
        if self.H is None:
            self.H = self.H0
            return
        dpos = pos - pos0
        dforces = forces - forces0
        batch_indices_flatten = self.optimizable.batch_indices.repeat_interleave(3)
        dg = torch.zeros_like(dforces)
        all_size = self.optimizable.elem_per_group

        for i in range(self.optimizable.batch_size):
            if self.H[i] is None:
                continue
            mask = (i==batch_indices_flatten)
            if torch.abs(dpos[mask]).max() < 1e-7:
                continue

            dg[mask] = self.H[i] @ dpos[mask]

        a = self._batched_dot_1d(dforces, dpos)
        b = self._batched_dot_1d(dpos, dg)

        for i in range(self.optimizable.batch_size):
            if self.H[i] is None:
                self.H[i] = torch.eye(3*all_size[i], device=self.device, dtype=torch.float64) * self.alpha
                continue
            mask = (i==batch_indices_flatten)
            if not self.optimizable.update_mask[i]:
                continue
            if torch.abs(dpos[mask]).max() < 1e-7:
                continue

            outer_force = torch.outer(dforces[mask], dforces[mask])
            outer_dg = torch.outer(dg[mask], dg[mask])
            self.H[i] -= outer_force / a[i] + outer_dg / b[i]

        

    def update_parallel(self, pos, forces, pos0, forces0):
        if self.H is None:
            self.H = self.H0
            return

        dpos = pos - pos0

        if torch.abs(dpos).max() < 1e-7:
            return

        dforces = forces - forces0
        cur_indices = self.optimizable.batch_indices.repeat_interleave(3)
        a = self._batched_dot_1d(dforces, dpos)
        # DONE: There is a bug using hstack.
        # dg = torch.hstack([self.H[i] @ dpos[cur_indices == i] for i in range(len(self.H))])
        # DONE: parallel this part
        # dg_list = [self.H[i] @ dpos[cur_indices == i] for i in range(len(self.H))]
        dg_list = [None] * len(self.H)
        streams = [torch.cuda.Stream() for _ in dg_list]
        for i, stream in zip(range(len(dg_list)), streams):
            with torch.cuda.stream(stream):
                dg_list[i] = self.H[i] @ dpos[cur_indices == i]

        torch.cuda.current_stream().synchronize()
        for stream in streams: 
            stream.synchronize()

        dg = torch.zeros_like(dforces)
        for i in torch.unique(cur_indices):
            mask = (cur_indices == i)
            dg[mask] = dg_list[i]
        b = self._batched_dot_1d(dpos, dg)

        # DONE: parallel this part
        for i, stream in zip(range(len(self.H)), streams):
            if not self.optimizable.update_mask[i]:
                continue
            with torch.cuda.stream(stream):
                outer_force = torch.outer(dforces[cur_indices==i], dforces[cur_indices==i])
                outer_dg = torch.outer(dg[cur_indices==i], dg[cur_indices==i])
                self.H[i] -= outer_force / a[i] + outer_dg / b[i]
        
        torch.cuda.current_stream().synchronize()
        for stream in streams: 
            stream.synchronize()


    def _batched_dot_2d(self, x: torch.Tensor, y: torch.Tensor):
        return scatter(
            (x * y).sum(dim=-1), self.optimizable.batch_indices, reduce="sum"
        )
    
    def _batched_dot_1d(self, x: torch.Tensor, y: torch.Tensor):
        return scatter(
            (x * y), self.optimizable.batch_indices.repeat_interleave(3), reduce="sum"
        ) 