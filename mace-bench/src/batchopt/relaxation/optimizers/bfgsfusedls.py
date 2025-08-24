"""
Copyright (c) 2025 Ma Zhaojia

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations
import logging
import torch
from torch_scatter import scatter
# from .linesearch_torch import LineSearchBatch
from ..optimizable import OptimizableBatch
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from datetime import datetime
import os
import math
import gc

class BFGSFusedLS:
    """
    Port of BFGSLineSearch from bfgslinesearch.py, adapted to PyTorch
    and batched operations, mirroring lbfgs_torch.py structure.
    """
    def __init__(
        self,
        optimizable_batch: OptimizableBatch,
        maxstep: float = 0.2,
        c1: float = 0.23,
        c2: float = 0.46,
        alpha: float = 10.0,
        stpmax: float = 50.0,
        device = 'cpu', 
        early_stop: bool = False,
        use_profiler: bool = False,
        profiler_log_dir: str = './log',
        profiler_schedule_config: dict = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.optimizable = optimizable_batch
        self.maxstep = maxstep
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.stpmax = stpmax
        self.nsteps = 0
        self.device = device
        self.force_calls = 0
        self.early_stop = early_stop
        self.use_profiler = use_profiler
        self.profiler_log_dir = profiler_log_dir
        self.profiler_schedule_config = profiler_schedule_config or {"wait": 48, "warmup": 1, "active": 1, "repeat": 8}
        self.dtype = dtype

        self.converge_indices_list = None

        # The information from the previous round is useful for the current round's calculations.
        ## These variables need to be update accroding to new input when eary stop is triggered.
        self.Hs = None
        self.r0 = None
        self.g0 = None
        self.p_list = [None] * self.optimizable.batch_size
        self.no_update_list = [False] * self.optimizable.batch_size
        self.ls_completed = [True] * self.optimizable.batch_size
        self.ls_batch = LineSearchBatch(self.optimizable.batch_indices, device="cpu", dtype=self.dtype)
        ## need to be recalculate when early stop is triggered
        self.forces = None
        self.energies = None
    
    def restart_from_earlystop(self, restart_indices, old_batch_indices):
        Hs_new = []
        r0_new = torch.zeros_like(self.optimizable.get_positions().reshape(-1), device=self.device)
        g0_new = torch.zeros_like(r0_new, device=self.device)
        p_list_new = []
        no_update_list_new = []
        ls_completed_new = []
            
        # collect the preserved historical info by old_indices
        for i, idx in enumerate(restart_indices):
            mask_old = (idx==old_batch_indices.repeat_interleave(3))
            mask = (i==self.optimizable.batch_indices.repeat_interleave(3))
            Hs_new.append(self.Hs[idx])
            p_list_new.append(self.p_list[idx])
            no_update_list_new.append(self.no_update_list[idx])
            ls_completed_new.append(self.ls_completed[idx])
            r0_new[mask] = self.r0[mask_old]
            g0_new[mask] = self.g0[mask_old]

        # append new info for new element in batch
        for i in range(len(Hs_new), self.optimizable.batch_size):
            # Hs_new.append(torch.eye(3 * self.optimizable.elem_per_group[i], device=self.device, dtype=torch.float64))
            Hs_new.append(None)
            p_list_new.append(None)
            no_update_list_new.append(False)
            ls_completed_new.append(True)

        self.Hs = Hs_new
        self.r0 = r0_new
        self.g0 = g0_new
        self.p_list = p_list_new
        self.no_update_list = no_update_list_new
        self.ls_completed = ls_completed_new
        self.forces = None
        self.energies = None
        self.ls_batch.restart_from_earlystop(restart_indices=restart_indices, batch_indices_new=self.optimizable.batch_indices)

    def step(self):
        optimizable = self.optimizable
        if self.forces is None:
            self.forces = optimizable.get_forces().to(self.device)
        r = optimizable.get_positions().reshape(-1).to(self.device)
        g = -self.forces.reshape(-1) / self.alpha
        p0_list = self.p_list
        self.update(r, g, self.r0, self.g0, p0_list)
        if self.energies is None:
            self.energies = self.func(r)

        for i in range(self.optimizable.batch_size):
            if self.ls_completed[i]:
                p = -torch.matmul(self.Hs[i], g[i==self.optimizable.batch_indices.repeat_interleave(3)])
                
                # Implement scaling for numerical stability with simpler calculation
                p_size = torch.sqrt((p**2).sum())
                min_size = torch.sqrt(self.optimizable.elem_per_group[i] * 1e-10)
                if p_size <= min_size:
                    p = p * (min_size / p_size)
                
                self.p_list[i] = p

        # ls_batch = LineSearchBatch(self.optimizable.batch_indices, device="cpu")
        continue_search = [not elem for elem in self.ls_completed]
        self.alpha_k_list, self.e_list, self.e0_list, self.no_update_list, self.ls_completed = self.ls_batch._linesearch_batch(
            self.func, self.fprime, r, self.p_list, g, self.energies, None,
            maxstep=self.maxstep, c1=self.c1, c2=self.c2, stpmax=self.stpmax, continue_search=continue_search
        )

        # reset device for linesearch result
        for i in range(self.optimizable.batch_size):
            if self.ls_completed[i]:
                self.alpha_k_list[i] = self.alpha_k_list[i].to(self.device)
                self.p_list[i] = self.p_list[i].to(self.device)
        
        dr_tensor = torch.zeros_like(r)


        for i in range(self.optimizable.batch_size):
            # if check_cache:
            #     mask = (i == self.optimizable.batch_indices.repeat_interleave(3))
            #     dr_tensor_all[mask] = self.alpha_k_list[i].to(self.device) * self.p_list[i].to(self.device)

            if not self.ls_completed[i]:
                continue
            if self.alpha_k_list[i] is None:
                raise RuntimeError("LineSearch failed!")
            
            mask = (i == self.optimizable.batch_indices.repeat_interleave(3))
            dr_tensor[mask] = self.alpha_k_list[i] * self.p_list[i]

        # if check_cache:
        #     cached_pos = optimizable.get_positions().reshape(-1).to(self.device) 
        #     update_pos = r + dr_tensor_all
        #     assert torch.allclose(update_pos, cached_pos), "dr_tensor_cached should be equal to dr_tensor"


        # TODO: get_forces/get_potential_energies will trigger compare_batch which is time-consuming
        forces_cache = optimizable.get_forces()
        energies_cache = self.optimizable.get_potential_energies() / self.alpha

        # update self.forces
        for i in range(self.optimizable.batch_size):
            if not self.ls_completed[i]:
                continue
            mask = (i == self.optimizable.batch_indices)
            self.forces[mask] = forces_cache[mask]
            self.energies[i] = energies_cache[i]

        optimizable.set_positions((r + dr_tensor).reshape(-1, 3))

        self.r0 = r
        self.g0 = g

    # @torch.compile
    def update(self, r, g, r0, g0, p0_list):
        all_sizes = self.optimizable.elem_per_group

        if self.Hs is None:
            self.Hs = [
                torch.eye(3 * sz, device=self.device, dtype=self.dtype)
                for sz in all_sizes
            ]
            return 

        dr = r - r0
        dg = g - g0

        for i in range(self.optimizable.batch_size):
            if self.Hs[i] is None:
                self.Hs[i] = torch.eye(3 * all_sizes[i], device=self.optimizable.device, dtype=self.dtype)
                continue
            if not self.ls_completed[i]:
                continue
            if self.no_update_list[i] is True:
                print('skip update')
                continue

            cur_mask = (i == self.optimizable.batch_indices.repeat_interleave(3))
            cur_g = g[cur_mask]
            cur_p0 = p0_list[i]
            cur_g0 = g0[cur_mask]
            cur_dg = dg[cur_mask]
            cur_dr = dr[cur_mask]

            if not (((self.alpha_k_list[i] or 0) > 0 and
                abs(torch.dot(cur_g, cur_p0)) - abs(torch.dot(cur_g0, cur_p0)) < 0) or False):
                continue

            try: 
                rhok = 1.0 / (torch.dot(cur_dg, cur_dr))
            except: 
                rhok = 1000.0
                print("Divide-by-zero encountered: rhok assumed large")
            if torch.isinf(rhok):
                rhok = 1000.0
                print("Divide-by-zero encountered: rhok assumed large")
            I = torch.eye(all_sizes[i]*3, device=self.device, dtype=self.dtype)
            A1 = I - cur_dr[:, None] * cur_dg[None, :] * rhok
            A2 = I - cur_dg[:, None] * cur_dr[None, :] * rhok
            self.Hs[i] = (torch.matmul(A1, torch.matmul(self.Hs[i], A2)) +
                    rhok * cur_dr[:, None] * cur_dr[None, :])


    # def update(self, r, g, r0, g0, p0_list):
    #     self.Is = [
    #         torch.eye(sz * 3, dtype=torch.float64, device=self.device)
    #         for sz in self.optimizable.elem_per_group
    #     ]

    #     # TODO: BFGS for loop 是不是在被打断之后需要重建这个 self.Hs?
    #     # TODO: 并且我们保存的上一次的r,g,r0,g0也被丢弃了
    #     if self.Hs is None:
    #         self.Hs = [
    #             torch.eye(3 * sz, device=self.optimizable.device, dtype=torch.float64)
    #             for sz in self.optimizable.elem_per_group 
    #         ]
    #         return
    #     else:
    #         dr = r - r0
    #         dg = g - g0

    #         for i in range(self.optimizable.batch_size):
    #             if not self.ls_completed[i]:
    #                 continue
    #             cur_mask = (i==self.optimizable.batch_indices.repeat_interleave(3))
    #             cur_g = g[cur_mask]
    #             cur_p0 = p0_list[i]
    #             cur_g0 = g0[cur_mask]
    #             cur_dg = dg[cur_mask]
    #             cur_dr = dr[cur_mask]

    #             if not (((self.alpha_k_list[i] or 0) > 0 and
    #                 abs(torch.dot(cur_g, cur_p0)) - abs(torch.dot(cur_g0, cur_p0)) < 0) or False):
    #                 break

    #             if self.no_update_list[i] is True:
    #                 print('skip update')
    #                 break

    #             try: 
    #                 rhok = 1.0 / (torch.dot(cur_dg, cur_dr))
    #             except: 
    #                 rhok = 1000.0
    #                 print("Divide-by-zero encountered: rhok assumed large")
    #             if torch.isinf(rhok):
    #                 rhok = 1000.0
    #                 print("Divide-by-zero encountered: rhok assumed large")
    #             A1 = self.Is[i] - cur_dr[:, None] * cur_dg[None, :] * rhok
    #             A2 = self.Is[i] - cur_dg[:, None] * cur_dr[None, :] * rhok
    #             self.Hs[i] = (torch.matmul(A1, torch.matmul(self.Hs[i], A2)) +
    #                     rhok * cur_dr[:, None] * cur_dr[None, :])



    def func(self, x):
        self.optimizable.set_positions(x.reshape(-1, 3).to(self.device))
        return self.optimizable.get_potential_energies() / self.alpha

    def fprime(self, x):
        self.optimizable.set_positions(x.reshape(-1, 3).to(self.device))
        
        self.force_calls += 1
        forces = self.optimizable.get_forces().reshape(-1)
        return - forces / self.alpha
    
    def run(self, fmax, maxstep, is_restart_earlystop=False, restart_indices=None, old_batch_indices=None):
        logging.info("Enter bfgsfusedlinesearch's main program.")
        self.fmax = fmax
        self.max_iter = maxstep

        if is_restart_earlystop:
            self.restart_from_earlystop(restart_indices, old_batch_indices)

        iteration = 0
        max_forces = self.optimizable.get_max_forces(apply_constraint=True)
        logging.info("Step   Fmax(eV/A)")

        # Run with profiler if enabled
        if self.use_profiler:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pid = os.getpid()
            with torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=self.profiler_schedule_config["wait"],
                    warmup=self.profiler_schedule_config["warmup"],
                    active=self.profiler_schedule_config["active"],
                    repeat=self.profiler_schedule_config["repeat"]
                ),
                on_trace_ready=tensorboard_trace_handler(self.profiler_log_dir, worker_name=f"BFGSLS_{pid}"),
                with_stack=True,
                profile_memory=True,
            ) as prof:
                # Main optimization loop with profiling
                while iteration < self.max_iter and not self.optimizable.converged(
                    forces=None, fmax=self.fmax, max_forces=max_forces, f_upper_limit=1e25,
                ):
                    if self.early_stop and iteration > 0:
                        self.converge_indices_list = self.optimizable.converge_indices_list
                        if len(self.converge_indices_list) > 0:
                            logging.info(f"Early stopping at iteration {iteration}")
                            break

                    logging.info(
                        f"{iteration} " + " ".join(f"{x:18.15g}" for x in max_forces.tolist())
                    )

                    self.step()
                    max_forces = self.optimizable.get_max_forces(apply_constraint=True, forces=self.forces)
                    iteration += 1
                    
                    # Step the profiler in each iteration
                    prof.step()
                
        else:
            # Original optimization loop without profiling
            while iteration < self.max_iter and not self.optimizable.converged(
                forces=None, fmax=self.fmax, max_forces=max_forces, f_upper_limit=1e25,
            ):
                if self.early_stop and iteration > 0:
                    self.converge_indices_list = self.optimizable.converge_indices_list
                    if len(self.converge_indices_list) > 0:
                        logging.info(f"Early stopping at iteration {iteration}")
                        break

                logging.info(
                    f"{iteration} " + " ".join(f"{x:18.15g}" for x in max_forces.tolist())
                )

                self.step()
                max_forces = self.optimizable.get_max_forces(apply_constraint=True, forces=self.forces)
                iteration += 1

        logging.info(
            f"{iteration} " + " ".join(f"{x:18.15g}" for x in max_forces.tolist())
        )

        # GPU memory usage as per nvidia-smi seems to gradually build up as
        # batches are processed. This releases unoccupied cached memory.
        torch.cuda.empty_cache()
        gc.collect()

        # set predicted values to batch
        for name, value in self.optimizable.results.items():
            setattr(self.optimizable.batch, name, value)

        self.nsteps = iteration

        if self.early_stop:
            self.converge_indices_list = self.optimizable.converge_indices_list
            return self.converge_indices_list
        else:
            return self.optimizable.converged(
                forces=None, fmax=self.fmax, max_forces=max_forces
            )

    def _batched_dot_2d(self, x: torch.Tensor, y: torch.Tensor):
        return scatter(
            (x * y).sum(dim=-1), self.optimizable.batch_indices, reduce="sum"
        )
    
    def _batched_dot_1d(self, x: torch.Tensor, y: torch.Tensor):
        return scatter(
            (x * y), self.optimizable.batch_indices.repeat_interleave(3), reduce="sum"
        )

# flake8: noqa
import math
import torch
import logging

pymin = min
pymax = max


class LineSearch:
    def __init__(self, xtol=1e-14, device='cpu', dtype=torch.float64):
        self.xtol = xtol
        self.task = 'START'
        self.device = device
        self.dtype = dtype
        self.isave = torch.zeros(2, dtype=torch.int64, device=self.device)
        self.dsave = torch.zeros(13, dtype=self.dtype, device=self.device)
        self.fc = 0
        self.gc = 0
        self.case = 0
        self.old_stp = 0
    
    def initialize(self, xk, pk, gfk, old_fval, old_old_fval,
                    maxstep=.2, c1=.23, c2=0.46, xtrapl=1.1, xtrapu=4.,
                    stpmax=50., stpmin=1e-8):
        # Scalar parameters can stay as Python scalars
        self.stpmin = stpmin
        self.stpmax = stpmax
        self.xtrapl = xtrapl
        self.xtrapu = xtrapu
        self.maxstep = maxstep
        
        # Move tensors to the device
        self.pk = pk.to(self.device)
        xk = xk.to(self.device)
        gfk = gfk.to(self.device)

        phi0 = old_fval
            
        
        # This dot product needs tensors
        derphi0 = torch.dot(gfk, self.pk).item()
        
        # Use Python math for scalar calculations
        self.dim = len(pk)
        self.gms = math.sqrt(self.dim) * maxstep
        
        alpha1 = 1.0
        self.no_update = False
        self.gradient = True
        
        self.steps = []
        return alpha1, phi0, derphi0

    def prologue(self, fval, gval, pk_tensor, alpha1):
        phi0 = fval
        derphi0 = torch.dot(gval, pk_tensor)
        self.old_stp = alpha1
        # TODO: self.no_update == True: break is needed to reimplemented.

        return phi0, derphi0

    def epilogue(self):
        pass

    def _line_search(self, func, myfprime, xk, pk, gfk, old_fval, old_old_fval,
                     maxstep=.2, c1=.23, c2=0.46, xtrapl=1.1, xtrapu=4.,
                     stpmax=50., stpmin=1e-8, args=()):
        self.stpmin = stpmin
        self.pk = pk.to(self.device)
        self.stpmax = stpmax
        self.xtrapl = xtrapl
        self.xtrapu = xtrapu
        self.maxstep = maxstep

        xk = xk.to(self.device)

        # Convert inputs to torch tensors if they're not already
        if not isinstance(old_fval, torch.Tensor):
            phi0 = torch.tensor(old_fval, dtype=self.dtype, device=self.device)
        else:
            phi0 = old_fval.to(self.device)
            
        # Ensure pk and gfk are torch tensors
        pk_tensor = torch.tensor(pk, dtype=self.dtype, device=self.device) if not isinstance(pk, torch.Tensor) else pk.to(self.device)
        gfk_tensor = torch.tensor(gfk, dtype=self.dtype, device=self.device) if not isinstance(gfk, torch.Tensor) else gfk.to(self.device)
        
        derphi0 = torch.dot(gfk_tensor, pk_tensor)
        self.dim = len(pk)
        self.gms = torch.sqrt(torch.tensor(self.dim, dtype=self.dtype, device=self.device)) * maxstep
        alpha1 = 1.
        self.no_update = False

        if isinstance(myfprime, tuple):
            fprime = myfprime[0]
            gradient = False
        else:
            fprime = myfprime
            newargs = args
            gradient = True

        fval = phi0
        gval = gfk_tensor
        self.steps = []

        while True:
            stp = self.step(alpha1, phi0, derphi0, c1, c2,
                            self.xtol,
                            self.isave, self.dsave)

            if self.task[:2] == 'FG':
                alpha1 = stp
                
                # Get function value and gradient
                x_new = xk + stp * pk_tensor
                fval = func(x_new).to(self.device)
                self.fc += 1
                
                gval = fprime(x_new).to(self.device)
                if gradient:
                    self.gc += 1
                else:
                    self.fc += len(xk) + 1
                    
                phi0 = fval
                derphi0 = torch.dot(gval, pk_tensor)
                self.old_stp = alpha1
                
                if self.no_update == True:
                    break
            else:
                break

        if self.task[:5] == 'ERROR' or self.task[1:4] == 'WARN':
            stp = None  # failed
            
        return stp, fval.item(), old_fval.item() if isinstance(old_fval, torch.Tensor) else old_fval, self.no_update

    def step(self, stp, f, g, c1, c2, xtol, isave, dsave):
        if self.task[:5] == 'START':
            # Check the input arguments for errors.
            if stp < self.stpmin:
                self.task = 'ERROR: STP .LT. minstep'
            if stp > self.stpmax:
                self.task = 'ERROR: STP .GT. maxstep'
            if g >= 0:
                self.task = 'ERROR: INITIAL G >= 0'
            if c1 < 0:
                self.task = 'ERROR: c1 .LT. 0'
            if c2 < 0:
                self.task = 'ERROR: c2 .LT. 0'
            if xtol < 0:
                self.task = 'ERROR: XTOL .LT. 0'
            if self.stpmin < 0:
                self.task = 'ERROR: minstep .LT. 0'
            if self.stpmax < self.stpmin:
                self.task = 'ERROR: maxstep .LT. minstep'
            if self.task[:5] == 'ERROR':
                return stp

            # Initialize local variables.
            self.bracket = False
            stage = 1
            finit = f
            ginit = g
            gtest = c1 * ginit
            width = self.stpmax - self.stpmin
            width1 = width / .5
            
            # The variables stx, fx, gx contain the values of the step,
            # function, and derivative at the best step.
            # The variables sty, fy, gy contain the values of the step,
            # function, and derivative at sty.
            # The variables stp, f, g contain the values of the step,
            # function, and derivative at stp.
            stx = 0.0
            fx = finit
            gx = ginit
            sty = 0.0
            fy = finit
            gy = ginit
            stmin = 0.0
            stmax = stp + self.xtrapu * stp
            self.task = 'FG'
            self.save((stage, ginit, gtest, gx,
                       gy, finit, fx, fy, stx, sty,
                       stmin, stmax, width, width1))
            stp = self.determine_step(stp)
            return stp
        else:
            if self.isave[0] == 1:
                self.bracket = True
            else:
                self.bracket = False
            stage = self.isave[1]
            (ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin, stmax,
             width, width1) = self.dsave

            # If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
            # algorithm enters the second stage.
            ftest = finit + stp * gtest
            if stage == 1 and f < ftest and g >= 0.:
                stage = 2

            # Test for warnings.
            if self.bracket and (stp <= stmin or stp >= stmax):
                self.task = 'WARNING: ROUNDING ERRORS PREVENT PROGRESS'
            if self.bracket and stmax - stmin <= self.xtol * stmax:
                self.task = 'WARNING: XTOL TEST SATISFIED'
            if stp == self.stpmax and f <= ftest and g <= gtest:
                self.task = 'WARNING: STP = maxstep'
            if stp == self.stpmin and (f > ftest or g >= gtest):
                self.task = 'WARNING: STP = minstep'

            # Test for convergence.
            # if f <= ftest and abs(g) <= c2 * (- ginit):
            #     self.task = 'CONVERGENCE'
            if (f < ftest or math.isclose(f, ftest, rel_tol=1e-6, abs_tol=1e-5)) and (abs(g) < c2 * (- ginit) or math.isclose(abs(g), c2 * (- ginit), rel_tol=1e-6, abs_tol=1e-5)):
                self.task = 'CONVERGENCE'

            # Test for termination.
            if self.task[:4] == 'WARN' or self.task[:4] == 'CONV':
                self.save((stage, ginit, gtest, gx,
                           gy, finit, fx, fy, stx, sty,
                           stmin, stmax, width, width1))
                return stp

            stx, sty, stp, gx, fx, gy, fy = self.update(stx, fx, gx, sty,
                                                        fy, gy, stp, f, g,
                                                        stmin, stmax)

            # Decide if a bisection step is needed.
            if self.bracket:
                if abs(sty - stx) >= .66 * width1:
                    stp = stx + .5 * (sty - stx)
                width1 = width
                width = abs(sty - stx)

            # Set the minimum and maximum steps allowed for stp.
            if self.bracket:
                stmin = min(stx, sty)
                stmax = max(stx, sty)
            else:
                stmin = stp + self.xtrapl * (stp - stx)
                stmax = stp + self.xtrapu * (stp - stx)

            # Force the step to be within the bounds maxstep and minstep.
            stp = max(stp, self.stpmin) 
            stp = min(stp, self.stpmax)

            if (stx == stp and stp == self.stpmax and stmin > self.stpmax):
                self.no_update = True
                
            # If further progress is not possible, let stp be the best
            # point obtained during the search.
            if (self.bracket and stp < stmin or stp >= stmax) \
               or (self.bracket and stmax - stmin < self.xtol * stmax):
                stp = stx

            # Obtain another function and derivative.
            self.task = 'FG'
            self.save((stage, ginit, gtest, gx,
                       gy, finit, fx, fy, stx, sty,
                       stmin, stmax, width, width1))
            return stp

    def update(self, stx, fx, gx, sty, fy, gy, stp, fp, gp,
               stpmin, stpmax):
        sign = gp * (gx / abs(gx))

        # First case: A higher function value. The minimum is bracketed.
        # If the cubic step is closer to stx than the quadratic step, the
        # cubic step is taken, otherwise the average of the cubic and
        # quadratic steps is taken.
        if fp > fx:  # case1
            self.case = 1
            theta = 3. * (fx - fp) / (stp - stx) + gx + gp
            s = max(max(abs(theta), abs(gx)), abs(gp))
            gamma = s * math.sqrt((theta / s) ** 2. - (gx / s) * (gp / s))
            if stp < stx:
                gamma = -gamma
            p = (gamma - gx) + theta
            q = ((gamma - gx) + gamma) + gp
            r = p / q
            stpc = stx + r * (stp - stx)
            stpq = stx + ((gx / ((fx - fp) / (stp - stx) + gx)) / 2.) \
                * (stp - stx)
            if (abs(stpc - stx) < abs(stpq - stx)):
                stpf = stpc
            else:
                stpf = stpc + (stpq - stpc) / 2.

            self.bracket = True

        # Second case: A lower function value and derivatives of opposite
        # sign. The minimum is bracketed. If the cubic step is farther from
        # stp than the secant step, the cubic step is taken, otherwise the
        # secant step is taken.
        elif sign < 0:  # case2
            self.case = 2
            theta = 3. * (fx - fp) / (stp - stx) + gx + gp
            s = max(max(abs(theta), abs(gx)), abs(gp))
            gamma = s * math.sqrt((theta / s) ** 2 - (gx / s) * (gp / s))
            if stp > stx:
                gamma = -gamma
            p = (gamma - gp) + theta
            q = ((gamma - gp) + gamma) + gx
            r = p / q
            stpc = stp + r * (stx - stp)
            stpq = stp + (gp / (gp - gx)) * (stx - stp)
            if (abs(stpc - stp) > abs(stpq - stp)):
                stpf = stpc
            else:
                stpf = stpq
            self.bracket = True

        # Third case: A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative decreases.
        elif abs(gp) < abs(gx):  # case3
            self.case = 3
            # The cubic step is computed only if the cubic tends to infinity
            # in the direction of the step or if the minimum of the cubic
            # is beyond stp. Otherwise the cubic step is defined to be the
            # secant step.
            theta = 3. * (fx - fp) / (stp - stx) + gx + gp
            s = max(max(abs(theta), abs(gx)), abs(gp))

            # The case gamma = 0 only arises if the cubic does not tend
            # to infinity in the direction of the step.
            gamma = s * math.sqrt(max(0., (theta / s) ** 2 - (gx / s) * (gp / s)))
            if stp > stx:
                gamma = -gamma
            p = (gamma - gp) + theta
            q = (gamma + (gx - gp)) + gamma
            r = p / q
            if r < 0. and gamma != 0:
                stpc = stp + r * (stx - stp)
            elif stp > stx:
                stpc = stpmax
            else:
                stpc = stpmin
            stpq = stp + (gp / (gp - gx)) * (stx - stp)

            if self.bracket:
                # A minimizer has been bracketed. If the cubic step is
                # closer to stp than the secant step, the cubic step is
                # taken, otherwise the secant step is taken.
                if abs(stpc - stp) < abs(stpq - stp):
                    stpf = stpc
                else:
                    stpf = stpq
                if stp > stx:
                    stpf = min(stp + .66 * (sty - stp), stpf)
                else:
                    stpf = max(stp + .66 * (sty - stp), stpf)
            else:
                # A minimizer has not been bracketed. If the cubic step is
                # farther from stp than the secant step, the cubic step is
                # taken, otherwise the secant step is taken.
                if abs(stpc - stp) > abs(stpq - stp):
                    stpf = stpc
                else:
                    stpf = stpq
                stpf = min(stpmax, stpf)
                stpf = max(stpmin, stpf)

        # Fourth case: A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative does not decrease. If the
        # minimum is not bracketed, the step is either minstep or maxstep,
        # otherwise the cubic step is taken.
        else:  # case4
            self.case = 4
            if self.bracket:
                theta = 3. * (fp - fy) / (sty - stp) + gy + gp
                s = max(max(abs(theta), abs(gy)), abs(gp))
                gamma = s * math.sqrt((theta / s) ** 2 - (gy / s) * (gp / s))
                if stp > sty:
                    gamma = -gamma
                p = (gamma - gp) + theta
                q = ((gamma - gp) + gamma) + gy
                r = p / q
                stpc = stp + r * (sty - stp)
                stpf = stpc
            elif stp > stx:
                stpf = stpmax
            else:
                stpf = stpmin

        # Update the interval which contains a minimizer.
        if fp > fx:
            sty = stp
            fy = fp
            gy = gp
        else:
            if sign < 0:
                sty = stx
                fy = fx
                gy = gx
            stx = stp
            fx = fp
            gx = gp
            
        # Compute the new step.
        stp = self.determine_step(stpf)

        return stx, sty, stp, gx, fx, gy, fy

    def determine_step(self, stp):
        dr = stp - self.old_stp
        x = torch.reshape(self.pk.to(self.device), (-1, 3))
        steplengths = ((dr * x)**2).sum(1)**0.5
        maxsteplength = max(steplengths)
        if maxsteplength >= self.maxstep:
            dr *= self.maxstep / maxsteplength
        stp = self.old_stp + dr
        return stp

    def save(self, data):
        if self.bracket:
            self.isave[0] = 1
        else:
            self.isave[0] = 0
        self.isave[1] = data[0]
        self.dsave = data[1:]

class LineSearchBatch:

    def __init__(self, batch_indices, device='cpu', dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.batch_indices = batch_indices.to(self.device)
        self.batch_indices_flatten = self.batch_indices.repeat_interleave(3).to(self.device)
        self.batch_size = len(torch.unique(batch_indices))
        self.linesearch_list = [LineSearch(device=self.device, dtype=self.dtype) for _ in range(self.batch_size)]
        self.steps = [1.] * self.batch_size
        self.phi0_values = [None] * self.batch_size
        self.derphi0_values = [None] * self.batch_size

    def restart_from_earlystop(self, restart_indices, batch_indices_new):
        self.batch_indices = batch_indices_new.to(self.device)
        self.batch_indices_flatten = self.batch_indices.repeat_interleave(3).to(self.device)
        self.batch_size = len(torch.unique(batch_indices_new))

        linesearch_list_new = []
        steps_new = []
        phi0_values_new = []
        derphi0_values_new = []

        for i, idx in enumerate(restart_indices):
            linesearch_list_new.append(self.linesearch_list[idx])
            steps_new.append(self.steps[idx])
            phi0_values_new.append(self.phi0_values[idx])
            derphi0_values_new.append(self.derphi0_values[idx])

        for i in range(len(restart_indices), self.batch_size):
            linesearch_list_new.append(LineSearch(device=self.device))
            steps_new.append(1.)
            phi0_values_new.append(None)
            derphi0_values_new.append(None)

        self.linesearch_list = linesearch_list_new
        self.steps = steps_new
        self.phi0_values = phi0_values_new
        self.derphi0_values = derphi0_values_new
        
        

    def _linesearch_batch(self, func, myfprime, xk, pk, gfk, old_fval, old_old_fval,
                            maxstep=.2, c1=.23, c2=0.46, xtrapl=1.1, xtrapu=4.,
                            stpmax=50., stpmin=1e-8, continue_search=None, max_iter=15):
        if continue_search is None:
            self.linesearch_list = [LineSearch(device=self.device) for _ in range(self.batch_size)]
        else:
            assert len(continue_search) == self.batch_size
            for i in range(len(continue_search)):
                if not continue_search[i]:
                    self.linesearch_list[i] = LineSearch(device=self.device)
        
        if isinstance(xk, torch.Tensor):
            xk = xk.to(self.device)
        for i in range(len(pk)): 
            pk[i] = pk[i].to(self.device)
        if isinstance(gfk, torch.Tensor):
            gfk = gfk.to(self.device)
        if isinstance(old_fval, torch.Tensor):
            old_fval = old_fval.to(self.device)
        if isinstance(old_old_fval, torch.Tensor):
            old_old_fval = old_old_fval.to(self.device)


        # results for each batch element
        alpha_results = []
        e_result = []
        e0_result = []
        no_update_result = []

        # Initialize step sizes and line search state for each batch element
        completed = [False] * self.batch_size
        
        # Initialize iteration counter
        iter_count = 0

        # Initialize all line searches using the initialize method
        for i in range(self.batch_size):
            if continue_search[i]:
                continue

            ls = self.linesearch_list[i]
            mask = (i == self.batch_indices_flatten)
            
            # Use the initialize method to set up line search parameters
            alpha1, phi0, derphi0 = ls.initialize(
                xk[mask], pk[i], gfk[mask], old_fval[i], old_old_fval,
                maxstep, c1, c2, xtrapl, xtrapu, stpmax, stpmin
            )
            
            # Store the initialization values
            self.steps[i] = alpha1
            self.phi0_values[i] = phi0
            self.derphi0_values[i] = derphi0
        
        # Main optimization loop
        while True:
            # 1. step forward
            # logging.info(f"step's input: alpha1: {torch.tensor([step.item() if isinstance(step, torch.Tensor) else step for step in self.steps])}")
            for i in range(self.batch_size):
                if completed[i]:
                    continue
                ls = self.linesearch_list[i]
                if ls.fc > max_iter:
                    completed[i] = True
                    logging.warning(f"LineSearchBatch[{i}] reached max_iter: {max_iter}")
                    continue
                stp = ls.step(self.steps[i], self.phi0_values[i], self.derphi0_values[i], 
                                c1, c2, ls.xtol, ls.isave, ls.dsave)
                if ls.task[:2] == 'FG':
                    self.steps[i] = stp
                else:
                    completed[i] = True
                        
            # 2. calculate new function value and gradient
            x_new_batch = torch.zeros_like(xk)
            for i in range(self.batch_size):
                mask = (i == self.batch_indices_flatten)
                x_new_batch[mask] = xk[mask] + self.steps[i] * pk[i]
            f_batch = func(x_new_batch).to(self.device)
            g_batch = myfprime(x_new_batch).to(self.device)

            # 3. update function value and gradient
            for i in range(self.batch_size):
                ls = self.linesearch_list[i]
                mask = (i == self.batch_indices_flatten)
                if ls.task[:2] == 'FG':
                    # Update function value and gradient
                    f_val = f_batch[i:i+1]
                    g_val = g_batch[mask]
                    ls.fc += 1
                    phi0, derphi0 = ls.prologue(f_val, g_val, pk[i], self.steps[i])
                    # logging.info(f"phi0, derphi0: {phi0}, {derphi0}")
                    self.phi0_values[i] = phi0
                    self.derphi0_values[i] = derphi0 # TODO: why we put the derphi0 here instead of set it inside the LineSearch class?
                    if ls.no_update:
                        completed[i] = True
                else:
                    completed[i] = True

            iter_count += 1
            logging.info(f"LineSearchBatch iter: {iter_count}: alpha: {torch.tensor([step.item() if isinstance(step, torch.Tensor) else step for step in self.steps])}")
            if any(completed):
                break

            # 4. set a linesearch upper limit
            # if iter_count > max_iter:
            #     for i in range(self.batch_size):
            #         completed[i] = True
            #     logging.warning(f"LineSearchBatch reached max_iter: {max_iter}")
            #     break
        
        # Collect results
        for i in range(self.batch_size):
            ls = self.linesearch_list[i]
            if ls.task[:5] == 'ERROR' or ls.task[1:4] == 'WARN':
                stp = torch.tensor(1., device=self.device)
            else:
                stp = self.steps[i] if isinstance(self.steps[i], torch.Tensor) else torch.tensor(self.steps[i], device=self.device)
                
            alpha_results.append(stp)
            e_result.append(self.phi0_values[i].item() if self.phi0_values[i] is not None else None)
            e0_result.append(old_fval[i].item() if isinstance(old_fval[i], torch.Tensor) else old_fval[i])
            no_update_result.append(ls.no_update)

        logging.info(f"LineSearchBatch finished in {iter_count} iterations. \
                     LineSearch Status: {[stat for stat in completed]}")
        
        return alpha_results, e_result, e0_result, no_update_result, completed
